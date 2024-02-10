# Reading Compression
Today I read a benchmark paper on prining neural networks by [Blalock et al.](https://arxiv.org/pdf/2003.03033.pdf) and more details on a specific method by [Han et al.](https://arxiv.org/abs/1510.00149). 
The first gave a good overview of the state of pruning in 2020, and my main takeaways are
- Pruning can be effective if you have decided on architectural and implementational details, but these are also important
- The specific pruning method is not too important, as other factors can have similar impacts on e.g. accuracy drops. 
However, pruning _does_ work in the sesne that models become more performant for the memory/FLOPs/energy they require.

Next, the paper by Han et al. combines pruning, quantization and Huffman-coding (on the set of unique retained weights). 
Pruning gives roughly a 10x memory reduction, quantization another 3-4x and Huffman coding about 30% (1.5x).
In light of Blalock et al., I realize that these results could depend on nusances, but since it is hard to benchmark pruning methods overall and since pruning methods in general tend to work, I might go ahead with this method for an initial baseline. 

Indeed, if I decide on a model and dataset now, then pruning + quantization + potentially Huffman-coding + potentially SVD (last post) will likely be helpful for running these models on my PC. 

### Knowledge distillation
Another exciting outlook would be to apply knowledge distillation (e.g. [Hinton et al.](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/44873.pdf)) to a smaller version of whatever architecure (e.g. Voxgraf) that I decide on, before performing the other optimization. 
In the case of Voxgraf, the most realistic situation is possibly to make a low-resolution mock-up of the generator architecture.
I will have to see how hard that is on my hardware.
