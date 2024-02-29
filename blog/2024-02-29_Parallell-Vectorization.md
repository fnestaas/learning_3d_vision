# Made stuff run much faster
I realized that we can parallellize the gaussians being processed if we do alpha blending cleverly after processing, and I implemented this, which lead to about a 2x speedup.
Then I moved from for loops to numpy vectorized computations, which brought rendering times to below 20 seconds on higher resolutions (400x400) on my machine. 
That used to take 10 minutes.

However, 15 seconds is not blazing fast, and I am wondering how to speed it up further.
I tried checking when we have reached sufficient alpha saturation, but the added time of checking that in vectorized computations seems to defeat the benefit.
Interestingly, the runtimes do not change much when I change the resolution of the picture, indicating that what takes time might be more the "overhead" to calling the function, and not the function calls themselves. 
I will have to investigate that.

For now, I might make this repo public soon, seeing as I have reached my first goal of making running these models feasible on my PC.

### Not sure
Whether there are bugs with the Camera or Gaussian class - some rendering is strange when changing viewing positions.

### Put on hold:
The speedups planned last time.
E.g. the IterativeImageSegmenter would require evaluating some Gaussians multiple times, so I figured this paralellization approach would be faster. 
Thus I also did not fix any potential bugs associated with it.
