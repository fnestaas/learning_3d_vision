# More Vectorization

Implemented `MultiGaussian`, which tries to compute matrices for many Gaussians at once instead of looping over individual Gaussians to do so.
The reason is that since rendering in a higher resolution now seems just about as fast as rendering in lower resolutions ((200, 200) and (400, 400) images, that is), I suspect that the major slowdown is not computing the values of the pixels, but looping over and getting information from individual Gaussians.
Unfortunately, I was not able to test this yet, but started work on `plot_opacity_multi`, which should do that.

For rendering without looping, I used blending over to find a matrix recursion equation, which allowed me to find the values for alpha after rendering every Gaussian analytically.
I suppose this has been done before, but I thought it was neat. 
Hopefully I will find a way to index such that we can exploit the speedup of having such an explicit solution.
