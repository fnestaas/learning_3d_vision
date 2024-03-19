# Running the code
1. Download the Gaussian Splat model [using this link](https://media.reshot.ai/models/plush_sledge/output.zip), and save it to `{Path of your Choice} / debug/point_cloud/iteration_30000/point_cloud.ply` 
1. create the file `.env` in the root directory of the code, and add `MODEL_PATH="{Path of your Choice}"` (same path as above).
1. Run `pip install -r requirements.txt` (dockerized version in the works)
1. Run the code in `scribbles.ipynb` ot `visualizer.py` (more interactive)

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
- Update: we are at 2.5 seconds to generate an image, but I will now stop writing blogs after every session. 

## Plan
The next goal is to implement training, experimenting first with SfM.
This should be feasible because we can render low resolution images in ~1 second. 
