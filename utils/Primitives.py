import numpy as np
from utils.Camera import Camera
from utils.constants import *
from scipy import spatial
import warnings
import random
from typing import List, Optional
from dataclasses import dataclass

from typing import List, Dict 
from utils.Camera import Camera 
import time 
import multiprocessing as mp 
import functools

## the below implementation has/had some parts which I do not fully understand, and I suspect there may be optimizations to be made too.
class Gaussian:
    def __init__(
            self,
            pos: np.array,
            scale: np.array, # standard deviations of each independent gaussian to use here. They are rotated to give the correct covariance
            rot: np.array, # quaternions
            opacity,
            sh,
            camera: Camera=None
        ):
        self.pos = np.array(pos)
        self.scale = np.array(scale)
        self.max_scale = self.scale.max()
        # Initialize scipy Quaternion from rot (s, x, y, z)
        self.rot = spatial.transform.Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]])
        self.opacity = opacity[0] # why?
        self.sh = np.array(sh)
        self.cov3D = self.compute_cov3d()
        self.camera = camera
        self.vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

    def compute_cov3d(self):
        cov3D = np.diag(self.scale**2)
        cov3D = self.rot.as_matrix().T @ cov3D @ self.rot.as_matrix()
        return cov3D

    def get_pos_cam(self, camera: Camera) -> np.array:
        """Get the 4d position of the gaussian, as viewed by the camera"""
        view_mat = self.get_view_matrix(camera) # camera.get_view_matrix() # world space to view space # TODO: better to save this matrix and update it when necessary
        g_pos_w = np.append(self.pos, 1.0)
        g_pos_cam = view_mat @ g_pos_w
        return g_pos_cam

    def _update_camera(self, camera: Camera):
        if camera:
            if not self.camera or np.linalg.norm(self.camera.position - camera.position) > 1e-4 or (self.camera.w - camera.w)**2 > 1e-4 or (self.camera.h - camera.h)**2 > 1e-4:
                self.camera = camera

    def get_view_matrix(self, camera: Camera=None):
        self._update_camera(camera)
        if not hasattr(self, 'view_matrix') or camera: self.view_matrix = self.camera.get_view_matrix()
        return self.view_matrix
        # return camera.get_view_matrix()

    def get_projection_matrix(self, camera: Camera=None):
        self._update_camera(camera)
        if not hasattr(self, 'projection_matrix') or camera: self.projection_matrix = self.camera.get_view_matrix()
        return self.view_matrix
        # return camera.get_projection_matrix()

    def get_cov2d(self, camera: Camera=None, ) -> np.ndarray:
        """Get 2d covariance in ndc"""
        # g_pos_cam = self.get_pos_cam(camera)
        view_matrix = self.get_view_matrix(camera)
        if camera is None: camera = self.camera
        [htan_fovx, htan_fovy, focal] = camera.get_htanfovxy_focal() # I guess this has to do with perspective rendering, but not sure

        # Implementation inspired by https://www.songho.ca/opengl/gl_projectionmatrix.html
        # We are assuming for all objects that l=r, t=u (see link for definition)
        n = focal # I admit this is slightly hacky, but we project to the focus plane, and I think the other implementation also had some issues
        f = -focal
        h = camera.h
        w = camera.w
        J = np.array(
            [
                [n/w * 2, 0., 0., 0.],  # TODO: w or h?
                [0., n/h * 2, 0., 0.],  # TODO: w or h?
                [0., 0., -(n+f) / (f-n), 0.-2*n*f/(f - n)],
                [0., 0., -1., 0.]
            ]
        )[:3, :3] # ignore w
        W = view_matrix[:3, :3].T
        T = W @ J
        cov = T.T @ self.cov3D @ T

        return cov[:2, :2]

    def get_depth(self, camera: Camera):
        """Get the perceived distance to the objects, as seen from the camera"""
        view_matrix = camera.get_view_matrix()

        position4 = np.append(self.pos, 1.0) # last component is w factor, with which we divide to account for objects looking smaller in perspective projections
        g_pos_view = view_matrix @ position4
        depth = g_pos_view[2]
        return depth

    def get_optimal_bb(self, camera: Camera, thresh: float=3., conic=None):
        # "inverse" of covariance - can be used to find active areas for each gaussian
        if conic is None:
            cov2d = self.get_cov2d(camera) # covariance

            det = np.linalg.det(cov2d)

            det_inv = 1.0 / max([1e-14, det]) # instead of comparing det == 0. as was done earlier
            cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]] # unique elements of covariance matrix
            conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv])

        # optimal bounding box: maximize x and y with the constraint -(ksi-mu)^T Sigma^-1 (ksi - mu) = -2*thresh^2, where ksi=(x, y), mu = 0 (centered Gaussian) and Sigma is the covariance
        # we can use a trick by observing that when y is maximal, and the above constraint holds, then there is a unique value of x such that the equation is satisfied.
        # satisfying the constraint means
        # x = -p/2 +- sqrt(p^2 - 4q), with p = 2y c[1]/c[0], q = (y^2 c[2] / c[0] - 2thresh^2) , c = conic
        # So p^2 - 4q must be 0, meaning
        # y^2 c1^2 / c0^2 - (y^2 c2 - 2t^2)/c0 = 0
        # y^2 (c1^2 / c0^2 - c2/c0) = - 2t^2/c0
        # y = +- sqrt(2) t/(sqrt(- c1^2 / c0 + c2))

        c0, c1, c2 = conic
        y_opt = thresh / (np.sqrt(-c1**2 / c0 + c2)) # * np.sqrt(2)
        x_opt = thresh / (np.sqrt(-c1**2 / c2 + c0)) # * np.sqrt(2)
        bboxsize_cam = np.array([x_opt, y_opt]) # bounds on coordinate values of 3 sigma ellipse level set. NOT ndc
        return bboxsize_cam

    def get_fast_bb(self, thresh: float=3.):
        bboxsize_cam = thresh*np.array([self.max_scale] * 2) # not pretty, but fast and correct. Could solve optimization problem based on conv2d instead (fix a level set and find the maximum value of x/y)
        return bboxsize_cam

    def get_conic(self, camera):
        cov2d = self.get_cov2d(camera) # covariance

        det = np.linalg.det(cov2d)

        det_inv = 1.0 / max([1e-14, det]) # instead of comparing det == 0. as was done earlier
        cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]] # unique elements of covariance matrix
        conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]) # TODO: convert from NDC to pixel space
        return conic


    def get_conic_and_bb(self, camera: Camera, thresh:float=3., optimal:bool=False):
        """Get conic and other bounding boxes. Is this implementation sound?"""
        cov2d = self.get_cov2d(camera) # covariance

        det = np.linalg.det(cov2d)

        det_inv = 1.0 / max([1e-14, det]) # instead of comparing det == 0. as was done earlier
        cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]] # unique elements of covariance matrix
        conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]) # TODO: convert from NDC to pixel space

        if not optimal:
            bboxsize_cam = self.get_fast_bb(thresh)
        else:
            bboxsize_cam = self.get_optimal_bb(camera, thresh, conic)
        # wh = np.array([camera.w, camera.h])
        bboxsize_ndc = bboxsize_cam# np.clip(bboxsize_cam, -1, 1)

        vertices = self.vertices # this is actually unneccessary... TODO
        bboxsize_cam = np.multiply(vertices, bboxsize_cam) # assuming bboxsize_cam has bounds on x and y for the ellipse, this bounds ul, ur, ll, lr. In principle, that's unneccesary as it uses 8, and not 4, coords (TODO)
        g_pos_ndc = self.get_pos_ndc(camera)

        bbox_ndc = np.multiply(vertices, bboxsize_ndc) + g_pos_ndc[:2]
        bbox_ndc = np.hstack((bbox_ndc, np.zeros((vertices.shape[0],2), dtype=self.dtype))) # concatenate bbox with "dummy" zeros
        bbox_ndc[:,2:4] = g_pos_ndc[2:4] # replace the dummy zeros above with z and w from clip position of gaussian. TODO: This is strange, in light of the way we define bboxsize_cam

        return conic, bboxsize_cam, bbox_ndc

    def get_pos_ndc(self, camera: Camera):
        """ndc pos, it seems, based on code in get_conic_bb"""
        # compute g_pos_screen and gl_position
        view_matrix = self.get_view_matrix(camera)# camera.get_view_matrix()
        projection_matrix = self.get_projection_matrix(camera) # camera.get_projection_matrix() # MAKE NDC

        position4 = np.append(self.pos, 1.0)
        g_pos_view = view_matrix @ position4
        g_pos_screen = projection_matrix @ g_pos_view # aka g_pos_clip
        g_pos_screen = g_pos_screen / g_pos_screen[3]
        return g_pos_screen

    def get_color(self, dir) -> np.ndarray:
        """Samples spherical harmonics to get color for given view direction"""
        # TODO: review
        c0 = self.sh[0:3]   # f_dc_* from the ply file)
        color = SH_C0 * c0

        shdim = len(self.sh)

        if shdim > 3:
            # Add the first order spherical harmonics
            c1 = self.sh[3:6]
            c2 = self.sh[6:9]
            c3 = self.sh[9:12]

            x = dir[0]
            y = dir[1]
            z = dir[2]
            color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3

        if shdim > 12:
            c4 = self.sh[12:15]
            c5 = self.sh[15:18]
            c6 = self.sh[18:21]
            c7 = self.sh[21:24]
            c8 = self.sh[24:27]

            (xx, yy, zz) = (x * x, y * y, z * z)
            (xy, yz, xz) = (x * y, y * z, x * z)

            color = color +	SH_C2_0 * xy * c4 + \
                SH_C2_1 * yz * c5 + \
                SH_C2_2 * (2.0 * zz - xx - yy) * c6 + \
                SH_C2_3 * xz * c7 + \
                SH_C2_4 * (xx - yy) * c8

        if shdim > 27:
            c9 = self.sh[27:30]
            c10 = self.sh[30:33]
            c11 = self.sh[33:36]
            c12 = self.sh[36:39]
            c13 = self.sh[39:42]
            c14 = self.sh[42:45]
            c15 = self.sh[45:48]

            color = color + \
                SH_C3_0 * y * (3.0 * xx - yy) * c9 + \
                SH_C3_1 * xy * z * c10 + \
                SH_C3_2 * y * (4.0 * zz - xx - yy) * c11 + \
                SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c12 + \
                SH_C3_4 * x * (4.0 * zz - xx - yy) * c13 + \
                SH_C3_5 * z * (xx - yy) * c14 + \
                SH_C3_6 * x * (xx - 3.0 * yy) * c15

        color += 0.5
        return np.clip(color, 0.0, 1.0)


class MultiGaussian:
    """Class that takes many Gaussians and uses vectorized methods in place of the Gaussian methods
    TODO: consider making it possible to compute alphas as a method in MultiGaussian. Then this can be done in parallell
    with computing and sorting depths
    """
    def __init__(
            self,
            camera: Optional[Camera]=None,
            gaussians: Optional[List[Gaussian]]=None,
            poss: Optional[np.array]=None,
            scales: Optional[np.array]=None, # standard deviations of each independent gaussian to use here. They are rotated to give the correct covariance
            rots: Optional[np.array]=None, # quaternions
            opacities: Optional[np.array]=None,
            shs: Optional[np.array]=None,
            ids=None, # which ids to take from parent
            parent=None, # MultiGaussian containing all of the Gaussians
            dtype=np.float64
        ):
        self.parent = parent
        self.ids = ids
        self.dtype = dtype 

        if parent is None:
            if gaussians is not None:
                poss = np.stack([g.pos for g in gaussians], axis=0)
                scales = np.stack([g.scale for g in gaussians], axis=0)
                rots = np.stack([g.rot.as_matrix() for g in gaussians], axis=0)
                shs = np.stack([g.sh for g in gaussians], axis=0)
                opacities = np.stack([g.opacity for g in gaussians], axis=0)
                self.gaussians = gaussians # for debugging
            self.n_gaussians = len(poss)
            self.pos = poss.astype(dtype)
            self.scale = scales.astype(dtype)
            self.max_scale = scales.max(axis=-1)
            self.min_scale = scales.min(axis=-1)
            # Initialize scipy Quaternion from rot (s, x, y, z)
            if gaussians is None: self.rot = np.stack([spatial.transform.Rotation.from_quat(r).as_matrix().astype(dtype) for r in rots], axis=0)
            else: self.rot = rots.astype(dtype)
            self.opacity = opacities.astype(dtype)
            self.sh = shs.astype(dtype)
            self.cov3D = self.compute_cov3d().astype(dtype)
            self.camera = camera
            self.vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

        else:
            self.pos = parent.pos[ids]
            self.rot = parent.rot[ids]
            self.sh = parent.sh[ids]
            self.opacity = parent.opacity[ids]
            self.scale = parent.scale[ids]
            self.cov3D = parent.cov3D[ids]
            self.camera = parent.camera
            self.vertices = parent.vertices
            self.n_gaussians = len(self.pos)
            self.max_scale = parent.max_scale[ids]
            self.min_scale = parent.min_scale[ids]
            self.dtype = parent.dtype

    def filter(self, criterion):
        return MultiGaussian(parent=self, ids=criterion(self))

    def __len__(self):
        return self.n_gaussians

    def compute_cov3d(self):
        diag = np.apply_along_axis(np.diag, arr=self.scale**2, axis=-1)
        rot = self.rot
        cov = rot.transpose(0, 2, 1) @ diag @ rot
        return cov

    def get_pos_cam(self, camera: Camera) -> np.array:
        """Get "5d vector" of 4d positions of the gaussian, as viewed by the camera"""
        view_mat = self.get_view_matrix(camera) # camera.get_view_matrix() # world space to view space # TODO: better to save this matrix and update it when necessary
        g_pos_w = np.concatenate([self.pos, np.ones((len(self),1), dtype=self.dtype)], axis=-1)
        g_pos_cam = np.apply_along_axis(lambda v: view_mat @ v, axis=-1, arr=g_pos_w) # view_mat @ g_pos_w
        return g_pos_cam

    def _update_camera(self, camera: Camera):
        if camera:
            if not self.camera or np.linalg.norm(self.camera.position - camera.position) > 1e-4 or (self.camera.w - camera.w)**2 > 1e-4 or (self.camera.h - camera.h)**2 > 1e-4:
                self.camera = camera

    def get_view_matrix(self, camera: Camera=None):
        # self._update_camera(camera)
        # if not hasattr(self, 'view_matrix') or camera: self.view_matrix = self.camera.get_view_matrix().astype(self.dtype)
        # return self.view_matrix
        if camera is None: camera = self.camera
        return camera.get_view_matrix()

    def get_projection_matrix(self, camera: Camera=None):
        # self._update_camera(camera)
        # if not hasattr(self, 'projection_matrix') or camera: self.projection_matrix = self.camera.get_view_matrix().astype(self.dtype)
        # return self.view_matrix
        if camera is None: camera = self.camera
        return camera.get_projection_matrix()

    def get_cov2d(self, camera: Camera=None, ) -> np.ndarray:
        """Get 2d covariance in ndc"""
        # g_pos_cam = self.get_pos_cam(camera)
        if camera is None: camera = self.camera
        view_matrix = self.get_view_matrix(camera)
        [htan_fovx, htan_fovy, focal] = camera.get_htanfovxy_focal() # I guess this has to do with perspective rendering, but not sure

        # Implementation inspired by https://www.songho.ca/opengl/gl_projectionmatrix.html
        # We are assuming for all objects that l=r, t=u (see link for definition)
        n = focal # I admit this is hacky, but we project to the focus plane, and I think the other implementation also had some issues
        f = -focal
        h = camera.h
        w = camera.w
        J = np.array(
            [
                [n/w * 2, 0., 0., 0.],  # TODO: w or h?
                [0., n/h * 2, 0., 0.],  # TODO: w or h?
                [0., 0., -(n+f) / (f-n), 0.-2*n*f/(f - n)],
                [0., 0., -1., 0.]
            ],
            dtype=self.dtype
        )[:3, :3] # ignore w
        W = view_matrix[:3, :3].T
        T = W @ J
        cov = T.T @ self.cov3D @ T

        return cov[:, :2, :2]

    def get_depth(self, camera: Camera):
        """Get the perceived distance to the objects, as seen from the camera"""
        view_matrix = self.get_view_matrix(camera)

        position4 = np.concatenate([self.pos, np.ones((len(self),1), dtype=self.dtype)], axis=-1) # last component is w factor, with which we divide to account for objects looking smaller in perspective projections
        g_pos_view = np.apply_along_axis(lambda v: view_matrix @ v, axis=-1, arr=position4) # view_mat @ g_pos_wview_matrix @ position4
        depth = g_pos_view[:, 2]
        return -depth # WHY??? TODO

    def get_optimal_bb(self, camera: Camera, thresh: float=3., conic=None):
        # "inverse" of covariance - can be used to find active areas for each gaussian
        if conic is None:
            cov2d = self.get_cov2d(camera) # covariance

            det = np.linalg.det(cov2d)# .astype(self.dtype)

            det_inv = 1.0 / np.maximum(1e-14, det) # instead of comparing det == 0. as was done earlier
            # cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]] # unique elements of covariance matrix
            # conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv])
            cov = [cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]]# (len(self), 3) matrices
            conic = [cov[:, 2] * det_inv, -cov[:, 1] * det_inv, cov[:, 0] * det_inv]

        # optimal bounding box: maximize x and y with the constraint -(ksi-mu)^T Sigma^-1 (ksi - mu) = -2*thresh^2, where ksi=(x, y), mu = 0 (centered Gaussian) and Sigma is the covariance
        # we can use a trick by observing that when y is maximal, and the above constraint holds, then there is a unique value of x such that the equation is satisfied.
        # satisfying the constraint means
        # x = -p/2 +- sqrt(p^2 - 4q), with p = 2y c[1]/c[0], q = (y^2 c[2] / c[0] - 2thresh^2) , c = conic
        # So p^2 - 4q must be 0, meaning
        # y^2 c1^2 / c0^2 - (y^2 c2 - 2t^2)/c0 = 0
        # y^2 (c1^2 / c0^2 - c2/c0) = - 2t^2/c0
        # y = +- sqrt(2) t/(sqrt(- c1^2 / c0 + c2))

        c0, c1, c2 = conic # all of shape (len(self), )
        y_opt = thresh / (np.sqrt(-c1**2 / c0 + c2)) # * np.sqrt(2)
        x_opt = thresh / (np.sqrt(-c1**2 / c2 + c0)) # * np.sqrt(2)
        bboxsize_cam = np.stack([x_opt, y_opt], axis=-1) # bounds on coordinate values of 3 sigma ellipse level set. NOT ndc
        return bboxsize_cam

    def get_fast_bb(self, thresh: float=3.):
        bboxsize_cam = thresh*np.stack([self.max_scale] * 2, axis=-1)
        return bboxsize_cam

    def bb_cam2bb_ndc(self, bboxsize_cam, camera):
        bboxsize_ndc = bboxsize_cam

        vertices = np.tile(self.vertices, (len(self), 1, 1))
        bboxsize_cam = np.multiply(vertices, bboxsize_cam.reshape(len(self), 1, 2)) # assuming bboxsize_cam has bounds on x and y for the ellipse, this bounds ul, ur, ll, lr. In principle, that's unneccesary as it uses 8, and not 4, coords (TODO)
        g_pos_ndc = self.get_pos_ndc(camera)

        bbox_ndc = np.multiply(vertices, bboxsize_ndc.reshape(len(self), -1, 2)) + g_pos_ndc[:, :2].reshape(len(self), -1, 2)
        return bboxsize_cam, bbox_ndc

    def get_conic(self, camera):
        cov2d = self.get_cov2d(camera) 

        det = np.linalg.det(cov2d).astype(self.dtype)

        det_inv = 1.0 / np.maximum(1e-14, det) # instead of comparing det == 0. as was done earlier
        cov = [cov2d[:, 0,0], cov2d[:, 0,1], cov2d[:, 1,1]] # unique elements of covariance matrix
        conic = [cov[2]*det_inv, -cov[1]*det_inv , cov[0]*det_inv]
        return conic 

    def get_conic_and_bb(self, camera: Camera, thresh:float=3., optimal:bool=False, bboxsize_cam:np.ndarray=None):
        """Get conic and other bounding boxes. Is this implementation sound?"""
        conic = self.get_conic(camera)
        if not optimal:
            bboxsize_cam = self.get_fast_bb(thresh)
        else:
            bboxsize_cam = self.get_optimal_bb(camera, thresh, conic)
        bboxsize_cam, bbox_ndc = self.bb_cam2bb_ndc(bboxsize_cam, camera)
        return np.stack(conic, axis=-1), bboxsize_cam, bbox_ndc

    def get_pos_ndc(self, camera: Camera):
        """ndc pos, it seems, based on code in get_conic_bb"""
        # compute g_pos_screen and gl_position
        view_matrix = self.get_view_matrix(camera)# camera.get_view_matrix()
        projection_matrix = self.get_projection_matrix(camera) # camera.get_projection_matrix() # MAKE NDC

        position4 = np.concatenate([self.pos, np.ones((len(self),1), dtype=self.dtype)], axis=-1)
        g_pos_view = np.apply_along_axis(lambda v: view_matrix @ v, axis=-1, arr=position4) 
        g_pos_screen = np.apply_along_axis(lambda v: projection_matrix @ v, axis=-1, arr=g_pos_view) # aka g_pos_clip
        g_pos_screen = g_pos_screen[..., :3] / g_pos_screen[:, 3:4] # divide by w (3:4 so that shape of divisor is correct)
        return g_pos_screen

    def get_color(self, dir) -> np.ndarray:
        """Samples spherical harmonics to get color for given view direction"""
        # TODO: review
        c0 = self.sh[:, 0:3]   # f_dc_* from the ply file)
        color = SH_C0 * c0

        # shdim = self.sh.shape[-1]

        # if shdim > 3:
        #     # Add the first order spherical harmonics
        #     c1 = self.sh[:, 3:6]
        #     c2 = self.sh[:, 6:9]
        #     c3 = self.sh[:, 9:12]

        #     x = dir[0]
        #     y = dir[1]
        #     z = dir[2]
        #     color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3

        # if shdim > 12:
        #     c4 = self.sh[:, 12:15]
        #     c5 = self.sh[:, 15:18]
        #     c6 = self.sh[:, 18:21]
        #     c7 = self.sh[:, 21:24]
        #     c8 = self.sh[:, 24:27]

        #     (xx, yy, zz) = (x * x, y * y, z * z)
        #     (xy, yz, xz) = (x * y, y * z, x * z)

        #     color = color +	SH_C2_0 * xy * c4 + \
        #         SH_C2_1 * yz * c5 + \
        #         SH_C2_2 * (2.0 * zz - xx - yy) * c6 + \
        #         SH_C2_3 * xz * c7 + \
        #         SH_C2_4 * (xx - yy) * c8

        # if shdim > 27:
        #     c9 = self.sh[:, 27:30]
        #     c10 = self.sh[:, 30:33]
        #     c11 = self.sh[:, 33:36]
        #     c12 = self.sh[:, 36:39]
        #     c13 = self.sh[:, 39:42]
        #     c14 = self.sh[:, 42:45]
        #     c15 = self.sh[:, 45:48]

        #     color = color + \
        #         SH_C3_0 * y * (3.0 * xx - yy) * c9 + \
        #         SH_C3_1 * xy * z * c10 + \
        #         SH_C3_2 * y * (4.0 * zz - xx - yy) * c11 + \
        #         SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c12 + \
        #         SH_C3_4 * x * (4.0 * zz - xx - yy) * c13 + \
        #         SH_C3_5 * z * (xx - yy) * c14 + \
        #         SH_C3_6 * x * (xx - 3.0 * yy) * c15

        color += 0.5
        return np.clip(color, 0.0, 1.0)

    def get_depth_thresh(self, thresh: float, reach: float, alphas: np.ndarray, depths:np.array=None, camera:Camera=None):
        # this method is mainly intended for the case where all gaussians have high opacity and large min_scale
        # Render alpha fast (filling bounding box with a constant value rather than gaussian distr) and track
        # depth where we reach alpha>=thresh. We can then use this information to avoid rendering gaussians
        # that are too deep in the image to be visible
        if camera is None: camera = self.camera
        reached_depths = np.inf * np.ones(alphas.shape, dtype=self.dtype)
        if depths is None: depths = self.get_depth(camera)
        bboxsize_cam = np.stack([self.min_scale]*2, axis=-1)
        bboxsize_cam, bbox_ndc =self.bb_cam2bb_ndc(bboxsize_cam, camera)
        x_cam_1_, x_cam_2_, y_cam_1_, y_cam_2_, x1_, x2_, y1_, y2_, nx_, ny_, mask = self.get_plotting_coords(alphas, bbox_ndc=bbox_ndc, bboxsize_cam=bboxsize_cam)
        opacity_ = self.opacity[mask]
        scale_ = self.min_scale[mask]
        for i, (x_cam_1, x_cam_2, y_cam_1, y_cam_2, x1, x2, y1, y2, nx, ny, opacity, scale) in enumerate(zip(
            x_cam_1_,
            x_cam_2_,
            y_cam_1_,
            y_cam_2_,
            x1_,
            x2_,
            y1_,
            y2_,
            nx_,
            ny_,
            opacity_,
            scale_
        )):
            # y_cam, x_cam = np.meshgrid(np.linspace(y_cam_1, y_cam_2, ny), np.linspace(x_cam_1, x_cam_2, nx), indexing='ij')
            # x = np.sqrt(max([x_cam_1**2, x_cam_2**2]))
            x = x_cam_1 
            y = y_cam_1
            # y = np.sqrt(max([y_cam_1**2, y_cam_2**2]))
            power = (x + y)**2 / (2*(scale*reach) ** 2)
            val = opacity * np.exp(-power*2) # multiply power by 2 since we are using a quadratic bb 
            alphas[y1:y2, x1:x2] += (1-alphas[y1:y2, x1:x2]) * val 
            reached_depths[y1:y2, x1:x2] = np.minimum(
                reached_depths[y1:y2, x1:x2], 
                np.where(alphas[y1:y2, x1:x2] < thresh, np.inf, depths[i])
            )
        return reached_depths
         

    def get_plotting_coords(self, alphas: np.ndarray, bboxsize_cam, bbox_ndc):
        def scale_wh(tnsr: np.ndarray, h: int, w: int):
            # scale values of tnsr's -1st axis, which has length 2, from -1, 1 to resp. 0, h and 0, w
            # replace this: np.array([(points_ndc[0] + 1) * width_half, (1.0 - points_ndc[1]) * height_half])
            tnsr = np.stack([tnsr[:, :, 0]+1, 1-tnsr[:, :, 1]], axis=-1) # TODO: why doesn't 1+tnsr[:, :, 1] work?
            tnsr = tnsr * np.stack([np.ones(tnsr.shape[:-1], dtype=self.dtype)*h, np.ones(tnsr.shape[:-1], dtype=self.dtype)*w], axis=-1)
            return tnsr / 2
        gaussians = self
        h, w = alphas.shape[:2]
        bbox_screen = scale_wh(bbox_ndc, h, w)

        ul = bbox_screen[:, 0,:2] # Bounding box vertices
        ur = bbox_screen[:, 1,:2]
        ll = bbox_screen[:, 3,:2]

        y1_ = np.maximum(np.floor(ul[:, 1]), np.zeros(len(gaussians), dtype=int)).astype(int)
        x1_ = np.maximum(np.floor(ul[:, 0]), np.zeros(len(gaussians), dtype=int)).astype(int)

        y2_ = np.minimum(np.ceil(ll[:, 1]), alphas.shape[0]*np.ones(len(gaussians), dtype=int)).astype(int)
        x2_ = np.minimum(np.ceil(ur[:, 0]), alphas.shape[1]*np.ones(len(gaussians), dtype=int)).astype(int)

        mask = np.logical_and(x2_ > x1_, y2_ > y1_)
        x1_ = x1_[mask]
        x2_ = x2_[mask]
        y1_ = y1_[mask]
        y2_ = y2_[mask]

        nx_ = x2_ - x1_
        ny_ = y2_ - y1_

        # Extract out inputs for the gaussian
        coordxy = bboxsize_cam[mask]
        x_cam_1_ = coordxy[:, 0, 0]   # ul
        x_cam_2_ = coordxy[:, 1, 0]   # ur
        y_cam_1_ = coordxy[:, 1, 1]   # ur (y)
        y_cam_2_ = coordxy[:, 2, 1]   # lr
        return (
            x_cam_1_,
            x_cam_2_,
            y_cam_1_,
            y_cam_2_,
            x1_,
            x2_,
            y1_,
            y2_,
            nx_,
            ny_,
            mask
        )


    def render(self, bitmap: np.ndarray, alphas: np.ndarray, camera: Camera=None, depths=None, depth_map=None):
        """Vectorized version of the plot_opacity function of the original repo - we want to loop as little as possible for speed"""
        if camera is None: 
            camera = self.camera
        else: self._update_camera(camera)
        gaussians = self
        conic_, bboxsize_cam, bbox_ndc = gaussians.get_conic_and_bb(camera, optimal=False) # different bounding boxes (active areas for gaussian)
        
        x_cam_1_, x_cam_2_, y_cam_1_, y_cam_2_, x1_, x2_, y1_, y2_, nx_, ny_, mask = self.get_plotting_coords(alphas, bboxsize_cam, bbox_ndc)
        gaussians = self  
        opacity_ = gaussians.opacity[mask]

        # camera_dir = gaussians.pos - camera.position.reshape((1, *camera.position.shape))
        # camera_dir = camera_dir / np.linalg.norm(camera_dir, axis=-1) # normalized camera viewing direction
        # color = gaussians.get_color(camera_dir)
        color_ = gaussians.get_color(None)[mask] 
        # check_alpha = hasattr(self.parent, 'finished') # Too much overhead... 
        for i, (x_cam_1, x_cam_2, y_cam_1, y_cam_2, x1, x2, y1, y2, nx, ny, opacity, color, conic) in enumerate(zip(
            x_cam_1_,
            x_cam_2_,
            y_cam_1_,
            y_cam_2_,
            x1_,
            x2_,
            y1_,
            y2_,
            nx_,
            ny_,
            opacity_,
            color_,
            conic_
        )):
            # if not check_alpha or not np.all(self.parent.finished[y1:y2, x1:x2]): # slows the loop down too much
            if depth_map is not None and depths[i] > depth_map[y1:y2,x1:x2].max(): 
                # print('lol')
                continue
            A, B, C = conic
            y_cam, x_cam = np.meshgrid(np.linspace(y_cam_1, y_cam_2, ny), np.linspace(x_cam_1, x_cam_2, nx), indexing='ij')
            power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
            alpha_ = opacity * np.exp(power)
            tmp = ((1-alphas[y1:y2, x1:x2]) * alpha_).astype(np.float32)
            bitmap[y1:y2, x1:x2] = bitmap[y1:y2, x1:x2] + (tmp[..., np.newaxis] * color[0:3][np.newaxis, np.newaxis]).astype(np.float32)
            alphas[y1:y2, x1:x2] = alphas[y1:y2, x1:x2] + tmp
        # finished = alphas > .5 # too slow to be useful
        # if hasattr(self.parent, 'finished'):
        #     self.parent.finished = np.logical_or(self.parent.finished, finished)
        # else: self.parent.finished = finished
        
        return bitmap, alphas


class GaussianModel:
    """
    Same as above, just using gaussians etc as global attributes
    """

    def __init__(
            self,
            camera: Optional[Camera]=None,
            gaussians: Optional[List[Gaussian]]=None,
            poss: Optional[np.array]=None,
            scales: Optional[np.array]=None, # standard deviations of each independent gaussian to use here. They are rotated to give the correct covariance
            rots: Optional[np.array]=None, # quaternions
            opacities: Optional[np.array]=None,
            shs: Optional[np.array]=None,
            ids=None, # which ids to take from parent
            dtype=np.float64
        ):
        self.ids = ids
        self.dtype = dtype 

        if ids is None:
            if gaussians is not None:
                poss = np.stack([g.pos for g in gaussians], axis=0)
                scales = np.stack([g.scale for g in gaussians], axis=0)
                rots = np.stack([g.rot.as_matrix() for g in gaussians], axis=0)
                shs = np.stack([g.sh for g in gaussians], axis=0)
                opacities = np.stack([g.opacity for g in gaussians], axis=0)
                self.gaussians = gaussians # for debugging
            self.n_gaussians = len(poss)
            self.ids = np.arange(self.n_gaussians, dtype=int)
            GaussianModel.pos = poss.astype(dtype)
            GaussianModel.scale = scales.astype(dtype)
            self.max_scale = scales.max(axis=-1)
            self.min_scale = scales.min(axis=-1)
            # Initialize scipy Quaternion from rot (s, x, y, z)
            if gaussians is None: GaussianModel.rot = np.stack([spatial.transform.Rotation.from_quat(r).as_matrix().astype(dtype) for r in rots], axis=0)
            else: GaussianModel.rot = rots.astype(dtype)
            GaussianModel.opacity = opacities.astype(dtype)
            GaussianModel.sh = shs.astype(dtype)
            GaussianModel.cov3D = GaussianModel.compute_cov3d().astype(dtype)
            GaussianModel.camera = camera
            GaussianModel.vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

        else:
            self.ids = ids 
            self.max_scale = GaussianModel.scale[ids].max(axis=-1)
            self.min_scale = GaussianModel.scale[ids].min(axis=-1)
            self.n_gaussians = len(self.max_scale)

    def __len__(self):
        return self.n_gaussians

    def compute_cov3d():
        diag = np.apply_along_axis(np.diag, arr=GaussianModel.scale**2, axis=-1)
        rot = GaussianModel.rot
        cov = rot.transpose(0, 2, 1) @ diag @ rot
        return cov

    def get_pos_cam(self, camera: Camera) -> np.array:
        """Get "5d vector" of 4d positions of the gaussian, as viewed by the camera"""
        view_mat = GaussianModel.get_view_matrix(camera) # camera.get_view_matrix() # world space to view space # TODO: better to save this matrix and update it when necessary
        g_pos_w = np.concatenate([GaussianModel.pos[self.ids], np.ones((len(self),1), dtype=self.dtype)], axis=-1)
        g_pos_cam = np.apply_along_axis(lambda v: view_mat @ v, axis=-1, arr=g_pos_w) # view_mat @ g_pos_w
        return g_pos_cam

    def _update_camera(camera: Camera):
        if camera:
            if not GaussianModel.camera or np.linalg.norm(GaussianModel.camera.position - camera.position) > 1e-4 or (GaussianModel.camera.w - camera.w)**2 > 1e-4 or (GaussianModel.camera.h - camera.h)**2 > 1e-4:
                GaussianModel.camera = camera

    def get_view_matrix(camera: Camera=None):
        # self._update_camera(camera)
        # if not hasattr(self, 'view_matrix') or camera: self.view_matrix = self.camera.get_view_matrix().astype(self.dtype)
        # return self.view_matrix
        if camera is None: camera = GaussianModel.camera
        return camera.get_view_matrix()

    def get_projection_matrix(camera: Camera=None):
        # self._update_camera(camera)
        # if not hasattr(self, 'projection_matrix') or camera: self.projection_matrix = self.camera.get_view_matrix().astype(self.dtype)
        # return self.view_matrix
        if camera is None: camera = GaussianModel.camera
        return camera.get_projection_matrix()

    def get_cov2d(self, camera: Camera=None, ) -> np.ndarray:
        """Get 2d covariance in ndc"""
        # g_pos_cam = self.get_pos_cam(camera)
        if camera is None: camera = GaussianModel.camera
        view_matrix = GaussianModel.get_view_matrix(camera)
        [htan_fovx, htan_fovy, focal] = camera.get_htanfovxy_focal() # I guess this has to do with perspective rendering, but not sure

        # Implementation inspired by https://www.songho.ca/opengl/gl_projectionmatrix.html
        # We are assuming for all objects that l=r, t=u (see link for definition)
        n = focal # I admit this is hacky, but we project to the focus plane, and I think the other implementation also had some issues
        f = -focal
        h = camera.h
        w = camera.w
        J = np.array(
            [
                [n/w * 2, 0., 0., 0.],  # TODO: w or h?
                [0., n/h * 2, 0., 0.],  # TODO: w or h?
                [0., 0., -(n+f) / (f-n), 0.-2*n*f/(f - n)],
                [0., 0., -1., 0.]
            ],
            dtype=self.dtype
        )[:3, :3] # ignore w
        W = view_matrix[:3, :3].T
        T = W @ J
        cov = T.T @ GaussianModel.cov3D[self.ids] @ T

        return cov[:, :2, :2]

    def get_depth(self, camera: Camera):
        """Get the perceived distance to the objects, as seen from the camera"""
        view_matrix = GaussianModel.get_view_matrix(camera)

        position4 = np.concatenate([GaussianModel.pos[self.ids], np.ones((len(self),1), dtype=self.dtype)], axis=-1) # last component is w factor, with which we divide to account for objects looking smaller in perspective projections
        g_pos_view = np.apply_along_axis(lambda v: view_matrix @ v, axis=-1, arr=position4) # view_mat @ g_pos_wview_matrix @ position4
        depth = g_pos_view[:, 2]
        return -depth # WHY??? TODO

    def get_optimal_bb(self, camera: Camera, thresh: float=3., conic=None):
        # "inverse" of covariance - can be used to find active areas for each gaussian
        if conic is None:
            cov2d = self.get_cov2d(camera) # covariance

            det = np.linalg.det(cov2d)# .astype(self.dtype)

            det_inv = 1.0 / np.maximum(1e-14, det) # instead of comparing det == 0. as was done earlier
            # cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]] # unique elements of covariance matrix
            # conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv])
            cov = [cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]]# (len(self), 3) matrices
            conic = [cov[:, 2] * det_inv, -cov[:, 1] * det_inv, cov[:, 0] * det_inv]

        # optimal bounding box: maximize x and y with the constraint -(ksi-mu)^T Sigma^-1 (ksi - mu) = -2*thresh^2, where ksi=(x, y), mu = 0 (centered Gaussian) and Sigma is the covariance
        # we can use a trick by observing that when y is maximal, and the above constraint holds, then there is a unique value of x such that the equation is satisfied.
        # satisfying the constraint means
        # x = -p/2 +- sqrt(p^2 - 4q), with p = 2y c[1]/c[0], q = (y^2 c[2] / c[0] - 2thresh^2) , c = conic
        # So p^2 - 4q must be 0, meaning
        # y^2 c1^2 / c0^2 - (y^2 c2 - 2t^2)/c0 = 0
        # y^2 (c1^2 / c0^2 - c2/c0) = - 2t^2/c0
        # y = +- sqrt(2) t/(sqrt(- c1^2 / c0 + c2))

        c0, c1, c2 = conic # all of shape (len(self), )
        y_opt = thresh / (np.sqrt(-c1**2 / c0 + c2)) # * np.sqrt(2)
        x_opt = thresh / (np.sqrt(-c1**2 / c2 + c0)) # * np.sqrt(2)
        bboxsize_cam = np.stack([x_opt, y_opt], axis=-1) # bounds on coordinate values of 3 sigma ellipse level set. NOT ndc
        return bboxsize_cam

    def get_fast_bb(self, thresh: float=3.):
        bboxsize_cam = thresh*np.stack([self.max_scale] * 2, axis=-1)
        return bboxsize_cam

    def bb_cam2bb_ndc(self, bboxsize_cam, camera):
        bboxsize_ndc = bboxsize_cam

        vertices = np.tile(self.vertices, (len(self), 1, 1))
        bboxsize_cam = np.multiply(vertices, bboxsize_cam.reshape(len(self), 1, 2)) # assuming bboxsize_cam has bounds on x and y for the ellipse, this bounds ul, ur, ll, lr. In principle, that's unneccesary as it uses 8, and not 4, coords (TODO)
        g_pos_ndc = self.get_pos_ndc(camera)

        bbox_ndc = np.multiply(vertices, bboxsize_ndc.reshape(len(self), -1, 2)) + g_pos_ndc[:, :2].reshape(len(self), -1, 2)
        return bboxsize_cam, bbox_ndc

    def get_conic(self, camera):
        cov2d = self.get_cov2d(camera) 

        det = np.linalg.det(cov2d).astype(self.dtype)

        det_inv = 1.0 / np.maximum(1e-14, det) # instead of comparing det == 0. as was done earlier
        cov = [cov2d[:, 0,0], cov2d[:, 0,1], cov2d[:, 1,1]] # unique elements of covariance matrix
        conic = [cov[2]*det_inv, -cov[1]*det_inv , cov[0]*det_inv]
        return conic 

    def get_conic_and_bb(self, camera: Camera, thresh:float=3., optimal:bool=False, bboxsize_cam:np.ndarray=None):
        """Get conic and other bounding boxes. Is this implementation sound?"""
        conic = self.get_conic(camera)
        if not optimal:
            bboxsize_cam = self.get_fast_bb(thresh)
        else:
            bboxsize_cam = self.get_optimal_bb(camera, thresh, conic)
        bboxsize_cam, bbox_ndc = self.bb_cam2bb_ndc(bboxsize_cam, camera)
        return np.stack(conic, axis=-1), bboxsize_cam, bbox_ndc

    def get_pos_ndc(self, camera: Camera):
        """ndc pos, it seems, based on code in get_conic_bb"""
        # compute g_pos_screen and gl_position
        view_matrix = GaussianModel.get_view_matrix(camera)# camera.get_view_matrix()
        projection_matrix = GaussianModel.get_projection_matrix(camera) # camera.get_projection_matrix() # MAKE NDC

        position4 = np.concatenate([GaussianModel.pos[self.ids], np.ones((len(self),1), dtype=self.dtype)], axis=-1)
        g_pos_view = np.apply_along_axis(lambda v: view_matrix @ v, axis=-1, arr=position4) 
        g_pos_screen = np.apply_along_axis(lambda v: projection_matrix @ v, axis=-1, arr=g_pos_view) # aka g_pos_clip
        g_pos_screen = g_pos_screen[..., :3] / g_pos_screen[:, 3:4] # divide by w (3:4 so that shape of divisor is correct)
        return g_pos_screen

    def get_color(self, dir) -> np.ndarray:
        """Samples spherical harmonics to get color for given view direction"""
        # TODO: review
        c0 = self.sh[self.ids, :3]   # f_dc_* from the ply file)
        color = SH_C0 * c0
        color += 0.5
        return np.clip(color, 0.0, 1.0)

    def get_depth_thresh(self, thresh: float, reach: float, alphas: np.ndarray, depths:np.array=None, camera:Camera=None):
        # this method is mainly intended for the case where all gaussians have high opacity and large min_scale
        # Render alpha fast (filling bounding box with a constant value rather than gaussian distr) and track
        # depth where we reach alpha>=thresh. We can then use this information to avoid rendering gaussians
        # that are too deep in the image to be visible
        if camera is None: camera = GaussianModel.camera
        reached_depths = np.inf * np.ones(alphas.shape, dtype=self.dtype)
        if depths is None: depths = self.get_depth(camera)
        bboxsize_cam = np.stack([self.min_scale]*2, axis=-1)
        bboxsize_cam, bbox_ndc =self.bb_cam2bb_ndc(bboxsize_cam, camera)
        x_cam_1_, x_cam_2_, y_cam_1_, y_cam_2_, x1_, x2_, y1_, y2_, nx_, ny_, mask = self.get_plotting_coords(alphas, bbox_ndc=bbox_ndc, bboxsize_cam=bboxsize_cam)
        opacity_ = GaussianModel.opacity[self.ids][mask]
        scale_ = self.min_scale[mask]
        for i, (x_cam_1, x_cam_2, y_cam_1, y_cam_2, x1, x2, y1, y2, nx, ny, opacity, scale) in enumerate(zip(
            x_cam_1_,
            x_cam_2_,
            y_cam_1_,
            y_cam_2_,
            x1_,
            x2_,
            y1_,
            y2_,
            nx_,
            ny_,
            opacity_,
            scale_
        )):
            # y_cam, x_cam = np.meshgrid(np.linspace(y_cam_1, y_cam_2, ny), np.linspace(x_cam_1, x_cam_2, nx), indexing='ij')
            # x = np.sqrt(max([x_cam_1**2, x_cam_2**2]))
            x = x_cam_1 
            y = y_cam_1
            # y = np.sqrt(max([y_cam_1**2, y_cam_2**2]))
            power = (x + y)**2 / (2*(scale*reach) ** 2)
            val = opacity * np.exp(-power*2) # multiply power by 2 since we are using a quadratic bb 
            alphas[y1:y2, x1:x2] += (1-alphas[y1:y2, x1:x2]) * val 
            reached_depths[y1:y2, x1:x2] = np.minimum(
                reached_depths[y1:y2, x1:x2], 
                np.where(alphas[y1:y2, x1:x2] < thresh, np.inf, depths[i])
            )
        return reached_depths
         

    def get_plotting_coords(self, alphas: np.ndarray, bboxsize_cam, bbox_ndc):
        def scale_wh(tnsr: np.ndarray, h: int, w: int):
            # scale values of tnsr's -1st axis, which has length 2, from -1, 1 to resp. 0, h and 0, w
            # replace this: np.array([(points_ndc[0] + 1) * width_half, (1.0 - points_ndc[1]) * height_half])
            tnsr = np.stack([tnsr[:, :, 0]+1, 1-tnsr[:, :, 1]], axis=-1) # TODO: why doesn't 1+tnsr[:, :, 1] work?
            tnsr = tnsr * np.stack([np.ones(tnsr.shape[:-1], dtype=self.dtype)*h, np.ones(tnsr.shape[:-1], dtype=self.dtype)*w], axis=-1)
            return tnsr / 2
        h, w = alphas.shape[:2]
        bbox_screen = scale_wh(bbox_ndc, h, w)

        ul = bbox_screen[:, 0,:2] # Bounding box vertices
        ur = bbox_screen[:, 1,:2]
        ll = bbox_screen[:, 3,:2]

        y1_ = np.maximum(np.floor(ul[:, 1]), np.zeros(len(self), dtype=int)).astype(int)
        x1_ = np.maximum(np.floor(ul[:, 0]), np.zeros(len(self), dtype=int)).astype(int)

        y2_ = np.minimum(np.ceil(ll[:, 1]), alphas.shape[0]*np.ones(len(self), dtype=int)).astype(int)
        x2_ = np.minimum(np.ceil(ur[:, 0]), alphas.shape[1]*np.ones(len(self), dtype=int)).astype(int)

        mask = np.logical_and(x2_ > x1_, y2_ > y1_)
        x1_ = x1_[mask]
        x2_ = x2_[mask]
        y1_ = y1_[mask]
        y2_ = y2_[mask]

        nx_ = x2_ - x1_
        ny_ = y2_ - y1_

        # Extract out inputs for the gaussian
        coordxy = bboxsize_cam[mask]
        x_cam_1_ = coordxy[:, 0, 0]   # ul
        x_cam_2_ = coordxy[:, 1, 0]   # ur
        y_cam_1_ = coordxy[:, 1, 1]   # ur (y)
        y_cam_2_ = coordxy[:, 2, 1]   # lr
        return (
            x_cam_1_,
            x_cam_2_,
            y_cam_1_,
            y_cam_2_,
            x1_,
            x2_,
            y1_,
            y2_,
            nx_,
            ny_,
            mask
        )


    def render(self, bitmap: np.ndarray, alphas: np.ndarray, camera: Camera=None, depths=None, depth_map=None, subimg_size=10):
        # TODO: make this compute depths for each frame. That way we can use fewer gaussians per plot and stop computations fast
        def compute_frame_powers(ids, xmin, xmax, ymin, ymax):
            x_cam_1 = x_cam_1_[ids]
            x_cam_2 = x_cam_2_[ids]
            y_cam_1 = y_cam_1_[ids]
            y_cam_2 = y_cam_2_[ids]
            x1 = x1_[ids]
            x2 = x2_[ids]
            y1 = y1_[ids]
            y2 = y2_[ids]
            A = conic_[ids, 0].reshape((-1, 1, 1))
            B = conic_[ids, 1].reshape((-1, 1, 1))
            C = conic_[ids, 2].reshape((-1, 1, 1))
            # interpolate to get bounds and make linspace
            # TODO: define a different variable with slopes to make this faster
            # TODO: find a better way than working with bbs directly...
            xspace = np.linspace(x_cam_1 + (x_cam_2 - x_cam_1) * (xmin-x1) / (x2-x1), x_cam_2 + (x_cam_2 - x_cam_1) * (xmax-x2) / (x2-x1), xmax-xmin, endpoint=False).T
            yspace = np.linspace(y_cam_1 + (y_cam_2 - y_cam_1) * (ymin-y1) / (y2-y1), y_cam_2 + (y_cam_2 - y_cam_1) * (ymax-y2) / (y2-y1), ymax-ymin, endpoint=False).T
            x_cam = xspace.reshape(len(xspace), 1, -1)
            y_cam = yspace.reshape(len(yspace), -1, 1) # TODO
            power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
            return opacity_[ids].reshape((-1, 1, 1)) * np.exp(power)

        def compute_alphas(alphas_):
            # TODO: sth like istar = np.max(np.argmax(a > .99, axis=0)) + 1 # or np.nonzero(a.min(axis=1).min(axis=1)>.99, axis=0).min()
            # r = np.cumprod(1-a[:istar], axis=0)
            # helper = a[:istar] / r 
            # res = np.concatenate([np.cumsum(helper, axis=0) * r, np.ones(len(a) - istar)], axis=0)
            # return np.minimum(res, 1.) # still needed since the required depth depends on the coordinates
            a = np.concatenate([np.zeros((1, *alphas_.shape[1:])), alphas_], axis=0)
            r = np.cumprod(1-a, axis=0) # alpha_i = l_i + r_i * alpha_0 # TODO: if a == 1, we are done, and we can just fill the result with 1's thereafter. Here, we can check if that happens
            helper = (a / np.maximum(r, 0.)) 
            res = np.cumsum(helper, axis=0) * r 
            return np.minimum(res, 1.) # TODO: this is true, but can we avoid computation above using the fact that once alpha_ is 1, res is also 1?

        if camera is None: 
            camera = self.camera
        else: GaussianModel._update_camera(camera)
        s = subimg_size
        # TODO: possible to speed up since we are working with frames and not bbs directly now?
        # E.g. while sorting, compute bb's and use them to make an (nh, nw, len(self)) boolean array (depends on camera)
        # keeping track of which gaussians are needed in which frame given a camera
        # We can then set all entries to False when we have reached a certain depth too
        conic_, bboxsize_cam, bbox_ndc = self.get_conic_and_bb(camera, optimal=True) # different bounding boxes (active areas for gaussian)
        w, h = alphas.shape
        # TODO: e.g. computing nx_ is no longer necessary. Think about optimizations here -- in principle, we only 
        # need to know which gaussians are included, and can compute their required bounds based on the frame rather than
        # the 3-sigma bounds
        x_cam_1_, x_cam_2_, y_cam_1_, y_cam_2_, x1_, x2_, y1_, y2_, nx_, ny_, mask = self.get_plotting_coords(alphas, bboxsize_cam, bbox_ndc)
        opacity_ = GaussianModel.opacity[self.ids][mask]
        conic_ = conic_[mask]
        # slopes_x = (x_cam_2_ - x_cam_1_) / (x2_ - x1_) # TODO: potentially do this and use in compute_frame_powers, but check first if we can rely less on bb's, as that might make this unneccesary
        nh = h//subimg_size + int(h%subimg_size != 0)
        nw = w//subimg_size + int(w%subimg_size != 0)
        color_ = self.get_color(None)[mask] 
        frame_x = np.arange(nh+1) * s # nh+1 due to stacking in next line
        frame_x = np.stack([frame_x[:-1], np.minimum(frame_x[1:], h)], axis=-1)
        frame_y = np.arange(nw+1) * s 
        frame_y = np.stack([frame_y[:-1], np.minimum(frame_y[1:], w)], axis=-1)
        frame_ids = np.logical_and(
            # explanation for x1_: the newaxes for x1_ are because x1_ has the same value irrespective of which grid coordinates we are looking at. 
            # However, that's not true for frame_x, which must be reshaped so that its values matches the frame coordinates.
            np.logical_and(x1_[np.newaxis, np.newaxis, :] < frame_x[:, 1].reshape((-1, 1, 1)), x2_[np.newaxis, np.newaxis, :] >= frame_x[:, 0].reshape((-1, 1, 1)), ), # x is in range
            np.logical_and(y1_[np.newaxis, np.newaxis, :] < frame_y[:, 1].reshape((1, -1, 1)), y2_[np.newaxis, np.newaxis, :] >= frame_y[:, 0].reshape((1, -1, 1)), ), # y is in range
        ) # shape (s, s, len(self))
        for i in range(nh): # TODO: maybe replace with np.ndenumerate
            for j in range(nw):
                curr_ids = frame_ids[i][j]
                if not np.any(curr_ids): continue
                frame_xmin = frame_x[i][0]
                frame_ymin = frame_y[j][0]
                frame_xmax = frame_x[i][1]
                frame_ymax = frame_y[j][1]
                
                alphas_ = compute_frame_powers(curr_ids, frame_xmin, frame_xmax, frame_ymin, frame_ymax) # (n, s, s) sized array
                # compute alphas in this sub image analytically from alphas_
                alpha_stack = compute_alphas(alphas_) # (n+1, s, s) sized array -- 0th element is 0 (size (s, s))
                
                # compute bitmap using alpha_stack and colors
                bitmap[frame_ymin:frame_ymax, frame_xmin:frame_xmax] = (
                    color_[
                        curr_ids
                    ][:, np.newaxis, np.newaxis] * ((1-alpha_stack[:-1])*alphas_)[..., np.newaxis]
                ).sum(axis=0)
                alphas[frame_ymin:frame_ymax, frame_xmin:frame_xmax] = alpha_stack[-1]
        # TODO: update self.frame_max_depths, and use this variable for skipping other computations too
        return bitmap, alphas

    def render_old(self, bitmap: np.ndarray, alphas: np.ndarray, camera: Camera=None, depths=None, depth_map=None):
        """Vectorized version of the plot_opacity function of the original repo - we want to loop as little as possible for speed"""
        if camera is None: 
            camera = self.camera
        else: GaussianModel._update_camera(camera)
        conic_, bboxsize_cam, bbox_ndc = self.get_conic_and_bb(camera, optimal=True) # different bounding boxes (active areas for gaussian)
        
        x_cam_1_, x_cam_2_, y_cam_1_, y_cam_2_, x1_, x2_, y1_, y2_, nx_, ny_, mask = self.get_plotting_coords(alphas, bboxsize_cam, bbox_ndc)
        opacity_ = GaussianModel.opacity[self.ids][mask]
        n_bins = 40 # TODO: figure out how this scales with resolution
        bins = [[None for _ in range(n_bins)] for _ in range(n_bins)]
        bin_positions = np.zeros((len(x_cam_1_), 3), dtype=int)
        # compute alphas by binning on nx, ny
        xstep = (nx_.max()+1) / n_bins
        ystep = (ny_.max()+1) / n_bins
        for xbin in range(n_bins):
            xrange = (xbin*xstep, (xbin+1)*xstep)
            for ybin in range(n_bins):
                yrange = (ybin*ystep, (ybin+1)*ystep)
                # find ids with nx_, ny_ with the correct xrange and yrange
                curr_ids = np.logical_and(
                    np.logical_and(xrange[0]<= nx_, nx_<xrange[1]),
                    np.logical_and(yrange[0]<= ny_, ny_<yrange[1]),
                )
                if not curr_ids.sum(): continue
                bin_positions[curr_ids] = np.concatenate([np.array([[xbin, ybin] for _ in range(curr_ids.sum())]), np.arange(curr_ids.sum(), dtype=int).reshape((-1, 1))], axis=-1) # the bin ids and idx in that bin
                # make linspace up to xrange.max, yrange.max, adjust according to x_cam, y_cam
                xspace = np.linspace(x_cam_1_[curr_ids], x_cam_2_[curr_ids]*(int(xrange[1])+1)/nx_[curr_ids], int(xrange[1])+1).T
                yspace = np.linspace(y_cam_1_[curr_ids], y_cam_2_[curr_ids]*(int(yrange[1])+1)/ny_[curr_ids], int(yrange[1])+1).T
                conic = conic_[mask][curr_ids].reshape((-1, 3, 1, 1))
                x_cam = xspace.reshape((len(xspace), -1, 1))
                y_cam = yspace.reshape((len(xspace), 1, -1)) # TODO: = y_cam[:, np.newaxis]
                power = -(conic[:, 0]*x_cam**2 + conic[:, 2]*y_cam**2)/2.0 - conic[:, 1] * x_cam * y_cam
                # append to bins[xrange][yrange] 
                bins[xbin][ybin] = np.transpose(opacity_[curr_ids].reshape((curr_ids.sum(), 1, 1))*np.exp(power), (0, 2, 1))


        color_ = self.get_color(None)[mask] 
        for i, (x1, x2, y1, y2, color, bin_pos) in enumerate(zip(
            x1_,
            x2_,
            y1_,
            y2_,
            color_,
            bin_positions
        )):
            xbin, ybin, j = bin_pos[0], bin_pos[1], bin_pos[2]
            alpha_ = bins[xbin][ybin][j][:y2-y1, :x2-x1]
            tmp = ((1-alphas[y1:y2, x1:x2]) * alpha_).astype(np.float32)
            bitmap[y1:y2, x1:x2] = bitmap[y1:y2, x1:x2] + (tmp[..., np.newaxis] * color[0:3][np.newaxis, np.newaxis]).astype(np.float32)
            alphas[y1:y2, x1:x2] = alphas[y1:y2, x1:x2] + tmp
        
        return bitmap, alphas


class PrimitiveSet:
    """Set of primitives (e.g. gaussians)"""
    def __init__(
        self,
        primitives: List[Gaussian] # TODO: for many applications, it would be nice to use a generator instead
    ):
        self.items = primitives
        self.indices = list(range(len(primitives)))

    def sample_ids(self, N:int):
        """sample randomly from self"""
        return random.sample(self.indices, N)

    def __len__(self):
        return len(self.items )

    def __getitem__(self, idx):
        return self.items[idx]

    def sort(self, camera):
        self.indices = np.argsort([gau.get_depth(camera) for gau in self.items])

@dataclass
class PrimitiveSubset:
    """A class that keeps track of a subset of Primitive objects"""
    ids: List[int]
    parent: PrimitiveSet
    def __init__(
        self,
        parent: PrimitiveSet,
        ids: List[int]
    ):
        self.parent = parent
        self.indices = ids
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.parent[idx]
    def sample_ids(self, N: int): return random.sample(self.indices, N)

