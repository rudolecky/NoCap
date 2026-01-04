torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --model d12 \
  --output_dir logs \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --train_sequence_length 512 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 5500 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --curriculum_steps 1000 \
  --start_seq_len 64 \
  --log_wandb
  