import os
import sys
import uuid
import math
import csv
import glob
from dataclasses import dataclass
from time import gmtime, strftime
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib.pyplot as plt
from flash_attn import flash_attn_func

with open(sys.argv[0]) as f:
    code = f.read()

# Logging Utilities
class TrainingLogger:
    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.plot_path = os.path.join(self.log_dir, "loss_plot.png")
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "val_loss", "learning_rate", "time_s"])
        
        self.train_steps = []
        self.train_losses = []
        self.val_steps = []
        self.val_losses = []

    def log_metrics(self, step, train_loss=None, val_loss=None, lr=0.0, time=0.0):
        # Write to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # We log 'nan' if a value isn't present for this step to keep columns aligned
            # t_loss = train_loss if train_loss is not None else ""
            v_loss = val_loss if val_loss is not None else ""
            writer.writerow([step, v_loss, lr, time])

        # Update internal history for plotting
        if train_loss is not None:
            self.train_steps.append(step)
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_steps.append(step)
            self.val_losses.append(val_loss)

    def update_plot(self):
        # Switch to object-oriented syntax (fig, ax) for dual-axis control
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- LEFT AXIS (Training) ---
        color_train = 'tab:blue'
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Train Loss', color=color_train)
        
        if self.train_steps:
            # Raw training loss
            ax1.plot(self.train_steps, self.train_losses, label="Train Loss", 
                    alpha=0.6, color=color_train)
            
            # Smoothed training loss
            if len(self.train_losses) > 10:
                window = 10
                avg_losses = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax1.plot(self.train_steps[window-1:], avg_losses, 
                        label="Train Loss (Smoothed)", color="darkblue", linewidth=2)
        
        # Color the tick labels to match the line color for clarity
        ax1.tick_params(axis='y', labelcolor=color_train)
        ax1.grid(True, which="major", linestyle="--", alpha=0.5)

        # --- RIGHT AXIS (Validation) ---
        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        color_val = 'tab:red'
        ax2.set_ylabel('Val Loss', color=color_val)

        if self.val_steps:
            ax2.plot(self.val_steps, self.val_losses, label="Val Loss", 
                    marker='o', color=color_val, linewidth=2)
        
        ax2.tick_params(axis='y', labelcolor=color_val)

        # --- COMBINED LEGEND ---
        # Because we have two axes, we must manually gather lines and labels
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title("Training vs Validation Loss (Dual Scales)")
        plt.savefig(self.plot_path)
        
        # Explicitly close the figure object to free memory
        plt.close(fig)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :])
        
    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[:, :seq_len, :, :], self.sin_cached[:, :seq_len, :, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def get_alibi_slope(num_heads, device):
    x = (2 ** 8) ** (1 / num_heads)
    # Shape: (num_heads,) - 1D tensor is sufficient for flash_attn
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device, dtype=torch.float32)

class ALiBiFlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.kqv = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.rotary = Rotary(self.head_dim)
        self.register_buffer("alibi_slopes", get_alibi_slope(self.num_heads, 'cuda'))

    def forward(self, x: torch.Tensor):
        # Input shape: (Batch, Seq, Hidden)
        batch_size, seq_len, _ = x.shape

        qkv = self.kqv(x)
        qkv = qkv.view(batch_size, seq_len, 4, self.num_heads, self.head_dim)
        
        q, k, v, g_score = qkv.unbind(dim=2)
        q = rmsnorm(q)
        k = rmsnorm(k)
        
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
                
        if q.dtype == torch.float32:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            g_score = g_score.to(torch.bfloat16)

        # Flash Attention with ALiBi
        out = flash_attn_func(
            q, k, v, 
            dropout_p=0.0, 
            softmax_scale=None,
            causal=True,
            alibi_slopes=self.alibi_slopes
        )
        out = out * torch.sigmoid(g_score)

        out = out.reshape(batch_size, seq_len, -1)
        if out.dtype != self.proj.weight.dtype:
            out = out.to(self.proj.weight.dtype)
            
        return self.proj(out)
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 8 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(x * F.silu(gate))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = ALiBiFlashAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.rmsnorm1 = RMSNorm(config.n_embd)
        self.rmsnorm2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(self.rmsnorm1(x))
        x = x + self.mlp(self.rmsnorm2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class SIGRegLoss(nn.Module):
    """
    SIGReg (Sketched Isotropic Gaussian Regularization).
    Ref: https://github.com/rbalestr-lab/lejepa
    
    It projects embeddings onto random directions and enforces them to match
    a standard normal distribution using a characteristic function (CF) loss.
    """
    def __init__(self, n_embd, num_slices=256, knots=17):
        super().__init__()
        self.n_embd = n_embd
        self.num_slices = num_slices
        self.num_points = knots
        # Points t to evaluate the characteristic function (hyperparameter)
        self.register_buffer("t", torch.randn(knots))

    def forward(self, x):
        x_flat = x.view(-1, self.n_embd)
        _, D = x_flat.shape
        # Random Projections (Slicing)
        directions = torch.randn(D, self.num_slices, device=x.device, dtype=x.dtype)
        projections = x_flat @ F.normalize(directions, dim=0)

        # Compute Empirical Characteristic Function (CF)
        p = projections.unsqueeze(-1)
        args = p * self.t.view(1, 1, -1) 
        
        empirical_real = torch.cos(args).mean(dim=0)
        # empirical_imag = torch.sin(args).mean(dim=0)
        target_real = torch.exp(-0.5 * self.t**2).view(1, -1)

        loss_real = (empirical_real - target_real).pow(2)
        # loss_imag = empirical_imag.pow(2)
        statistic = loss_real * x.size(-2)
                
        return statistic.mean()

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying
        self.rmsnorm = RMSNorm(config.n_embd)
        # self.sigreg = SIGRegLoss(config.n_embd)
        
        print(f"Number of parameters: {self.get_num_params():,}")
        
    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params


    def forward(self, idx, targets=None, training=False, return_logits=True):
        b, t = idx.size()
        with torch.autocast(device_type='cuda' if idx.is_cuda else 'cpu', 
                           dtype=torch.bfloat16 if idx.is_cuda else torch.float32,
                           enabled=self.training):
            # forward the GPT model itself
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

            for block in self.transformer.h:
                x = block(x)
            x = self.rmsnorm(x)

            if targets is not None:
                # if training:
                #     sigreg_loss = self.sigreg(x) if self.training else None
                # else:
                #     sigreg_loss = 0.0
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )                
                # sigreg_lambda = getattr(self.config, 'sigreg_lambda', 0.02)
                # loss = ce_loss + (sigreg_lambda * sigreg_loss)                
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(
                    x[:, [-1], :]
                )  # note: using list [-1] to preserve the time dim
                loss = None

            # there are performance reasons why not returning logits is prudent, if not needed
            if not return_logits:
                logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, current_T=None):
        if current_T is None: current_T = self.T
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = buf.astype(np.int64)
        
        # Inputs and Targets placeholders
        x_batch = np.zeros((B, current_T), dtype=np.int64)
        y_batch = np.zeros((B, current_T), dtype=np.int64)
        
        for i in range(B):
            # Extract a single sequence from the buffer
            # Note: We take current_T + 1 to have input + target shift
            seq_start = i * T 
            # Slice to current curriculum length
            raw_seq = buf[seq_start : seq_start + current_T + 1]
            
            seq = raw_seq[:current_T + 1]
            x_batch[i] = seq[:-1]
            y_batch[i] = seq[1:]

        # Convert to tensor
        x = torch.from_numpy(x_batch).long()
        y = torch.from_numpy(y_batch).long()
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="d12",
        help="d12|d24|d36|d48",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    parser.add_argument(
        "--train_sequence_length", type=int, default=64, help="sequence length"
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="every how mant steps to evaluate val loss?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="how many batches of val to average?",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="every how many steps to save the checkpoint",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="log to wandb",
    )
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="path to .pt file to resume from")
    parser.add_argument("--curriculum_steps", type=int, default=0, help="Steps to reach full sequence length")
    parser.add_argument("--start_seq_len", type=int, default=128, help="Starting sequence length for curriculum")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert (
        args.grad_accumulation_steps % ddp_world_size == 0
    ), "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= (
        ddp_world_size  # each gpu does its fraction of the work
    )
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    if args.log_wandb and master_process:
        import wandb
        import datetime

        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2.py")
        wandb.save("run.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # set up a context manager following the desired dtype and device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    # init the model from scratch 528
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(
            vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768
        ),  # 124M GPT-2
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    torch._dynamo.config.optimize_ddp = False
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(
        model
    )  # NOTE: this might cause issues depending on your GPU, consider turning it off

    
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())
    
    start_step = 0
    
    if args.resume_checkpoint is not None:
        print0(f"Resuming training from {args.resume_checkpoint}")
        # Load on CPU first to avoid OOM, then move to GPU
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        
        # 1. Load Model Weights
        # strict=False is useful if you added new layers (like MTP heads) 
        # that weren't in the checkpoint, but use strict=True if structure is identical.
        raw_model.load_state_dict(ckpt['model'], strict=True) 
        
        # 2. Load Optimizer State
        # This restores the momentum buffers so the loss doesn't spike
        optimizer.load_state_dict(ckpt['optimizer'])
        
        # 3. Restore Step
        start_step = ckpt['step'] + 1
        print0(f"Resumed from step {start_step}")

    # create the logging directory if it does not exist
    if master_process  and args.output_dir:
        run_name = f"{args.model}_batch{B}_lr{args.learning_rate}_{strftime('%d-%m_%H:%M:%S', gmtime())}_{os.path.basename(__file__)}"
        # run_name = f"{args.model}_batch{B}_lr{args.learning_rate}_{uuid.uuid4().hex[:4]}"
        log_dir = os.path.join(args.output_dir, run_name)
        logger = TrainingLogger(log_dir, args.model)
        print0(f"Logging metrics to: {log_dir}")

    training_time = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training
    # for step in range(args.num_iterations + 1):
    for step in range(start_step, args.num_iterations + 1):
        last_step = step == args.num_iterations
        
        if args.curriculum_steps > 0 and step < args.curriculum_steps:
            # Linear increase
            progress = step / args.curriculum_steps
            target_T = int(args.start_seq_len + (args.train_sequence_length - args.start_seq_len) * progress)
    
            # Snap to the nearest multiple of 128 to keep compiler happy
            # This ensures we only compile graphs for T=128, 256, 384...
            curr_T = (math.floor(target_T / 64) * 64)
            
            # Clamp to max length
            curr_T = min(curr_T, args.train_sequence_length)
            curr_T = max(curr_T, args.start_seq_len)
        else:
            curr_T = args.train_sequence_length

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time += (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):  # always fiexed number of validation steps
                    with ctx:
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                logger.log_metrics(step, val_loss=val_loss.item(), time=training_time)
                logger.update_plot()

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss = model(x, y, training=True, return_logits=False)
                loss = (
                    loss / args.grad_accumulation_steps
                )  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch(current_T=curr_T)
            # backward pass
            loss.backward()

        train_loss /= (
            args.grad_accumulation_steps
        )  # average the loss over all micro steps

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 500 == 0:
            torch.cuda.empty_cache()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        approx_training_time = training_time + (time.perf_counter() - t0)
        if master_process:
            # logger.log_metrics(step, train_loss=train_loss.item(), lr=lr, time=training_time)
            # Update plot less frequently during training to save IO, e.g., every 10 steps
            if step % 10 == 0:
                logger.update_plot()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time}s | step_avg:{approx_training_time/(step+1)}ms"
        )
        # log to logile
        if master_process and (step > 0) and (step % args.save_every == 0 or last_step):
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), # Crucial for resuming training momentum!
                "model_args": args.__dict__,
                "step": step,
                "config": model_config, # Optional: save config to ensure architecture matches
            }
            # If you are using the Muon/AdamW split, you might need to save both:
            # "opt_muon": opt_muon.state_dict(), "opt_adam": opt_adam.state_dict()
            
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            print0(f"saving checkpoint to logs/{run_id}/ckpt_{step}.pt")
            torch.save(checkpoint, f"logs/{run_id}/ckpt_{step}.pt")

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()