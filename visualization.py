"""
Crude visualizer, which is to be improved
"""

import matplotlib as mpl  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
import numpy as np
from utils.Camera import Camera 
from utils.Primitives import GaussianModel
from utils.rendering import ParRenderer 
from utils.util_gau import load_ply 
from pathlib import Path 
from decouple import config
from tqdm import tqdm 
import time
import scipy as sp 
import threading

def render(ax, ):
    print(azim_new, elev_new)
    azim = ax.azim 
    elev = ax.elev 
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Calculate the distance from the eye to the center of the scene
    center_x = xlim[1]
    center_y = ylim[1]
    center_z = zlim[1]

    dist = ((center_x - 0) ** 2 + (center_y - 0) ** 2 + (center_z - 0) ** 2) ** 0.5

    azim_ = np.deg2rad(azim)  # Convert to radians

    # Define elevation angle (in radians)
    elev_ = np.deg2rad(elev)  # Convert to radians

    # Convert spherical coordinates to Cartesian coordinates
    x = dist*np.cos(elev_) * np.cos(azim_)
    y = dist*np.cos(elev_) * np.sin(azim_)
    z = dist*np.sin(elev_)
    position = (x+target[0], y+target[1], z+target[2])
    # up = np.array([up_x, up_y, up_z])
    up = np.array([0., 0., 1.])
    print(f'{position=}, {dist=}')
    camera = Camera(w, h, position=position, target=target, up=up)
    bitmap_parts = renderer.plot_model_par_multiprimitives(camera, n_threads=n_threads, skip=5000, K=None)
    # do alpha blending post-hoc
    bitmap = np.zeros((w, h, 3))
    alpha = np.zeros((w, h))
    time.sleep(.1)
    for bm, al in tqdm(bitmap_parts):
        # if stop_event.is_set(): return 
        tmp = (1-alpha)*al
        bitmap = bitmap + (1-alpha)[..., np.newaxis]*bm # tmp[..., np.newaxis]*bm
        alpha = alpha + tmp 
    assert isinstance(ax2, plt.Axes)
    ax2.cla()
    ax2.imshow(bitmap, vmin=0, vmax=1.0)
    
    plt.show(block=False)
        

N = 10
# random gaussians:
poss = np.concatenate([
    2*(np.random.random((N, 3))-.5), 
    np.zeros((3, 3))*.7, 
    # np.eye(3)*.7
    # 2*(np.random.random((3, 3))-.5), 
], axis=0)
scales = np.concatenate([
    np.ones((N, 3))*.03, 
    np.eye(3)*.3+np.ones((3, 3))*.05
], 
axis=0)
rots = np.concatenate([
    np.stack([np.random.random(4,)*2*np.pi for _ in range(N)], axis=0),
    np.concatenate([np.zeros((3, 3)), np.ones((3, 1)), ], axis=-1), 
], axis=0)
shs = np.concatenate([
    np.random.random((N, 3)),
    np.eye(3)
], axis=0)
opacities = np.ones((N+3, ))

gaussian_objects = load_ply(str(Path(config('MODEL_PATH'))/'debug/point_cloud/iteration_30000/point_cloud.ply'), dtype=np.float64, return_type='gm')


n_threads = 20
delay = .3
resol = 400

# change up direction to be z (as in matplotlib)
gaussian_objects.pos -= gaussian_objects.pos.mean(axis=0)[np.newaxis, :]
mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)# np.transpose(np.eye(3), axes=(0, 2, 1))
gaussian_objects.pos = gaussian_objects.pos @ mat
gaussian_objects.scale = gaussian_objects.scale @ mat
gaussian_objects.rot = mat @ gaussian_objects.rot @ mat.T
gaussian_objects.cov3D =  mat @ gaussian_objects.cov3D @ mat.T
    
(w, h) = (resol, resol)
renderer = ParRenderer(gaussian_objects)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
# fig2, ax2 = plt.subplots(1, 1)
ax2 = fig.add_subplot(1, 2, 2)
target = (0., 0., 0.)
# ax.scatter(poss[:, 0], poss[:, 1], poss[:, 2])
idx = np.random.choice(range(len(gaussian_objects),), 5000)
poss = gaussian_objects.pos[idx]
colors = [np.linalg.norm((1, 1, 1) - p) / (1-poss).max() for p in poss]
ax.scatter(poss[:, 0], poss[:, 1], poss[:, 2], alpha=.1, c=colors)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Show the figure, adjust it with the mouse
plt.show(block=False)

azim = ax.azim 
elev = ax.elev 
almost_eq = lambda x1, x2, y1, y2: np.abs(x1-y1) + np.abs(x2-y2) < 1e-3
stop_event = threading.Event()
while True:
    plt.pause(delay)
    azim_new = ax.azim 
    elev_new = ax.elev 
    if not almost_eq(azim, elev, azim_new, elev_new):
        azim, elev = azim_new, elev_new
        plt.pause(delay)
        azim_new = ax.azim 
        elev_new = ax.elev 
        if almost_eq(azim, elev, azim_new, elev_new): render(ax)

