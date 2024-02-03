# First Steps
While I am still very much in the conceptualization phase, I imagine that anything I do with 3D vision will be 
- hard on my laptop, which has no GPU, or
- have to be done using a cloud computing service that gives me access to GPUs.

But it could be fun to challenge myself to make models run on my laptop. 
An initial idea would be to try to make [VoxGRAF](https://github.com/autonomousvision/voxgraf) run as fast as possible. 

In any case, I need to learn more about making models run as fast as possible. 
While I do know that quantization and pruning are methods that are used to speed up inference in LLMs (GPTQ etc), I did not know the details until reading up on it today. 
Also today, I figured there must be some LORA-like way to distill large neural networks, and found a [blogpost](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5) describing model compression in exactly this spirit, which is very cool. 

A possible project as of now would be to make VoxGRAF run as fast as possible, using the different methods outlined in that blog post. 
Also, if I did not misunderstand, the low-rank approximations outlined there are actually _only_ matrix/tensor decompositions, not followed by any fine-tuning/calibration, which seems like a sub-optimal scheme to follow. 
As such, a potential pipeline would be to benchmark model performance on my laptop using
- regular models
- compressed (SVD/CP Decomposition)
- compressed + fine tuned
- quantized + pruned (as in [this paper](https://arxiv.org/pdf/1510.00149.pdf))
- compressed + finetuned + quantized + pruned.

All of the fine-tuning methods sound like they might require GPUs. 
