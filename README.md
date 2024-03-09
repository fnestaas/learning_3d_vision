# Running the code
1. Download the Gaussian Splat model [using this link](https://media.reshot.ai/models/plush_sledge/output.zip)
2. Run `pip install -r requirements.txt` (dockerized version in the works)
3. Run the code in `scribbles.ipynb`

# About
## Learning 3D Vision Topics
This repo is here to track my progress on learning a topic within 3D-vision; an area with which I have no experience. 
My progress is documented in the blog - there is a post for every time I have worked on this for the first six weeks.

## Summary
- Started with no knowledge about 3d vision.
- Looked for topics, read papers on 3D (aware) GANs, NERFs and Gaussian Splats (on which I decided)
- Found an [implementation of Gaussian Splats for CPUs](https://github.com/thomasantony/splat/blob/master/notes/00_Gaussian_Projection.ipynb), which I used as a starting point, with the goal of making Gaussian Splats run fast on CPU.
- As of writing: pictures that took more than 5 mins to generate earlier (depending on the resolution) now take 15-20 seconds on my system.
Planning to keep speeding this up, but wanted to make the repo public already.
- Update: we are at 5 seconds to generate an image, but I will now stop writing blogs after every session. 
