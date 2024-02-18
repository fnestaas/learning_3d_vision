# Scribbles
Today I started reading about [optimizations already made](https://aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/) to Gaussian Splats and thought of some myself.
The ones I thought of myself which I also tried to implement in the notebook `scribbles.ipynb` do two things;
1. Segment the 2D-picture we want to render in such a way that each segment is affected by equally many primitives (roughly)
1. Assign only the relevant primitives to each part of the image (so that we only loop over each Gaussian once to check if it is rendered in one part of the image, instead of looping over once for every sub-image, as is done now).
The latter I did not get around to because I had some troubles understanding the implementation I am working with, and for my next session, I will have to understand that implementation better. 
In any case, the parallellization from last session is now in the notebook.

## Other optimizations
- In the reference notebook, when they draw the gaussians, they iterate over a box of points which is guaranteed to contain the Gaussian they are after. 
However, it would be more efficient to, for each x, use the Gaussian pdf (which as far as I can tell is not normalized in this framework) to compute the values of y where the pdf reaches a threshold value. 
- Possibly something with 3D Morton Orders as in the first link in this blog.
- SVD/grouping primitives that are often computed at the same time to exploit efficient matrix multiplication algorithms under the hood. 

## Plan for next session
- Better understand 3d rendering, camera views, etc. 
The problem I am having currently is that I do not understand some of the choices made in my reference repo's code (mathematical and implementation-wise), and I do not understand the coordinate systems.
- See if the algorithms I implemented today actually work
