from typing import List, Dict 
from utils.Primitives import MultiGaussian, GaussianModel
from utils.Camera import Camera 
import time 
import numpy as np 
import multiprocessing as mp 
import functools

# define utility functions for parallell computation


class ParRenderer:
    gaussian_objects: MultiGaussian

    def __init__(self, gaussian_objects) -> None:
        ParRenderer.gaussian_objects = gaussian_objects

    def helper_multiprimitives(
            indices: List[int],
            bitmap,
            alphas,
            K=1000,
            depths=None,
            depth_map=None
        ):
        # Clarification: Use class variable ParRenderer.gaussian_objects
        if K is None:
            # bitmap, alphas = MultiGaussian(parent=ParRenderer.gaussian_objects, ids=indices).render(bitmap, alphas, depths=depths, depth_map=depth_map, camera=ParRenderer.camera)
            bitmap, alphas = GaussianModel(ids=indices).render(bitmap, alphas, depths=depths, depth_map=depth_map, camera=ParRenderer.camera)
        else:
            for ids in [indices[i*K:(i+1)*K] for i in range(len(indices) // K + int(len(indices) % K != 0))]:
                # bitmap, alphas = MultiGaussian(parent=ParRenderer.gaussian_objects, ids=ids).render(bitmap, alphas, depths=depths, depth_map=depth_map, camera=ParRenderer.camera)
                bitmap, alphas = GaussianModel(ids=ids).render(bitmap, alphas, depths=depths, depth_map=depth_map, camera=ParRenderer.camera)
        return bitmap, alphas


    def plot_model_par_multiprimitives(self, camera: Camera, n_threads:int=1, skip=10000, K=1000, depth_map=None):
        gaussian_objects = ParRenderer.gaussian_objects
        ParRenderer.camera = camera 
        """Parallellize image rendering using MultiGaussians.
        Requires blending the segments later.
        """
        print('Sorting the gaussians by depth')
        
        t0 = time.time()
        depths = gaussian_objects.get_depth(camera)
        indices = np.argsort(depths)
        t1 = time.time()
        print(f'sorted in time {t1-t0:.2f} s')
        w, h = camera.w, camera.h
        
        print('Plotting with', len(gaussian_objects), 'gaussians')
        t0 = time.time()
        bitmap = np.zeros((h, w, 3), np.float32)
        alphas = np.zeros((h, w), np.float32)
        gsns = (indices[i*skip:(i+1)*skip] for i in range(len(indices)//skip + int(len(indices)%skip != 0)))
        t1 = time.time()
        if depth_map is not None: depth_map = np.where(depth_map < np.inf, depth_map, depths.max())
        print(f'prepped in time {t1-t0:.2f} s')
        t0 = time.time()
        if n_threads > 1:
            with mp.Pool(n_threads) as pool:
                for r in pool.imap(
                    functools.partial(
                        ParRenderer.helper_multiprimitives,
                        bitmap=bitmap,
                        alphas=alphas, 
                        K=K, 
                        depths=depths,
                        # gaussian_objects=gaussian_objects,
                        depth_map=depth_map
                    ), 
                    gsns
                ): 
                    yield r 
        else:
            for g in gsns:
                yield ParRenderer.helper_multiprimitives(
                        indices=g,
                        # camera=camera,# None,
                        bitmap=bitmap,
                        alphas=alphas,
                        K=K, 
                        depths=depths,
                        # gaussian_objects=gaussian_objects,
                        depth_map=depth_map
                    )
        t1 = time.time()
        print(f'rendered in time {t1-t0:.2f} s')
        

