# Clean Cuts

Today I worked on the following
- understanding coordinate systems better using [this great guide](https://learnopengl.com/Getting-started/OpenGL)
- Using my understanding to change a few things in the implementation I was working with, which I did not undestand why the authors implemented the way they did. 
Also I implemented optimal bounding boxes for gaussian level sets instead of the bounds that were in place already
- Broke the splatter plotting function, but I'm sure I can fix that later
- Debugged the code from last time and implemented `IterativeImageSegmenter` (still in `scribbles.py`) by building on the `cut_image`function (which I implemented last time but was able to make run well today, using optimal bounding boxes)

Essentially I now have more knowledge and some classes which will allow me to increase computational efficiency in the future (as outlined earlier. 
In particular, I won't have to consider all primitives for each sub-image when rendering parts of the image in parallell).
