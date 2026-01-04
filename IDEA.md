# GPT-2 Benchmark



I initially approached GPT architecture modifications by exploring ambitious design directions, including recursive models, heterogeneous and homogeneous expert architectures, and sparse modeling techniques. In practice, these changes resulted in high validation losses and, in several cases, training instability or outright divergence.
A key shift was inspired by the [BabyLM Challenge](https://babylm.github.io/), which emphasizes training on limited data with compact models. This highlighted how substantially the optimal architectural choices differ between small-scale and large-scale language models. Additionally, some published architectures proved difficult to replicate reliably, which motivated a pivot toward more incremental modifications.
Due to hardware constraints, my ablation studies were not as rigorous as desired. Most experiments were therefore conducted with shorter-than-ideal training runs, limiting the statistical confidence of some conclusions.


## Successful Modifications and Additions
* **SwiGLU**
* **Training gradient norm**
* **Rotary embedding adjustment:**
Rotary embeddings are computed in FP32 for improved numerical accuracy and only then rounded to the nearest FP16 value. This replaces the standard approach of performing the computation directly in FP16 during the forward pass.

* **RMSNorm with Learnable Parameters:**
Incorporating learnable parameters into RMSNorm improved model stability and performance.

* **ALiBi Attention:**
ALiBi enabled effective training on shorter sequence lengths without a noticeable degradation in performance. This made it possible to start training with a sequence length of 64 and progressively increase it up to 512.

* **Combining Rotary Embeddings with ALiBi:**
In one experiment, rotary embeddings were unintentionally combined with ALiBi attention, despite this not being a standard practice. Surprisingly, this configuration led to an improvement in model accuracy.

* **Gating:**
Gating described in [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708). While gating substantially increased the parameter count, it proved effective especially when applied o the attention layers. Consequently, the final model restricts gating to attention, in line with the findings reported in the paper.

* **SIGReg Loss:**
Inspired by [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544
), originally applied to Vision Transformers, I introduced an additional training loss that regularizes the distribution of embeddings. Early results indicated an added benefit from using only the real part of the characteristic function. While the underlying cause of the improved convergence requires further investigation, an initial hypothesis is that it relates to opposing interactions between the imaginary component and RMSNorm.
> **Edit:** 
> *Longer and more thorough ablation runs revealed that, although SIGReg Loss initially improved convergence, it led to higher loss values later in training. As a result, SIGReg Loss was removed from the final model configuration. Nevertheless, it remains a promising mechanism and a potential lever for future research.*


## Unsuccesfull modifications
* **Xielu**
* **Heterogeneous and homogeneous experts**
* **Recursive models**
* **Sparse models**
* [**Multi token prediction**](https://babylm.github.io/Posters/Babies%20Look%20Ahead%20Multi-Token%20Prediction%20in%20Small%20LMs.pdf)
* [**Co4**](https://babylm.github.io/Posters/Single%20Layer%20Tiny%20Co4%20outpaces%20GPT%202%20and%20GPT-BERT.pdf)
