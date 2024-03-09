# More Vectorization Again
Today I had a short ish session, where I tried making the stuff from last session work. 
I made a compromise where in the `plot_opacity_multi` there is still a loop for plotting alphas and colors, but that still gave an overall 2x speedup to the rendering process. 
I.e. now, it takes roughly as long to compute the Gaussians' depth and sort by them, as it does to plot on a 400x400 grid.
The reason for the speedup is likely that we are computing many matrices (covariances and bounding boxes) related to the Gaussians as a 3D numpy array/tensor, which is faster than looping over the Gaussians and computing the relevant 2D arrays sequentially.

Earlier, it took about 10 minutes to do that plotting, and currently, it takes 4-5 seconds (+2-3 seconds for depth sorting, which is unchanged), so we are looking at a 120-150x speedup to the reference implementation for the rendering part, by using parallellization and numpy vectorization.

Started working on a solution where we don't need the for loop, and I'm curious to see whether it's appereciably faster or not. 
In contrast to parallellization and vectorization, the approach I have in mind is not as "simple", as it requires some math, as touched upon in the last entry (I will provide the full details if it works).

The next speedups will likely come from
- Finishing the above implementation
- never using python lists, e.g. loading the .ply files directly as a `MultiGaussian` object

When that is done, I will consider stopping writing these blog entries, since they take time away from implementation, and because now the last few entries have been on very similar topics.
Nonetheless, they do give an insight into how I think and work to solve problems like this.
