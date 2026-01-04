# Baby GPT Benchmark

## 1. Summary
> This model successfully reduced time to validation loss ≤ 3.3821 in 11.83 hours compared to 17.66 hours of the baseline.
That makes the model **49% better** compared to the baseline.

## 2. Configuration & Hyperparameters

### Training Hyperparameters
| Parameter | Baseline GPT-2 | Baby GPT |
| :--- | :--- | :--- |
| **batch_size** | 16 | 16 |
| **grad_accumulation_steps** | 32 | 32 |
| **sequence_length** | 1024 | 1024 |
| **train_sequence_length** | 1024 | 512 |
| **val_loss_every** | 128 | 128 |
| **val_batch_size** | 16 | 16 |
| **num_iterations** | 4768 | 5500 |
| **weight_decay** | 0.1 | 0.1 |
| **learning_rate** | 0.0018 | 0.0018 |
| **warmup_iters** | 256 | 256 |
| **warmdown_iters** | 1024 | 1024 |
| **curriculum_steps** | - | 1000 |
| **start_seq_len** | - | 64 |

### Hardware Configuration
|  |  |
| :--- | :--- |
| **CPU** | AMD Ryzen 7 7700X |
| **RAM** | 64 GB RAM DDR5 6600MHz |
| **GPU** | NVIDIA GeForce RTX 4060 Ti 16GB |


## 3. Training Metrics
### Quantitative Data Table
| Model | Validation loss | Training time (h) |
| :--- | :--- | :--- |
| Baseline GPT-2 | 3.3799 | 17.66 |
| Baby GPT | 3.3671 | 11.83 |
