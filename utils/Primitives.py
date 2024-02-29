import numpy as np 
from utils.Camera import Camera 
from utils.constants import * 
from scipy import spatial
import warnings
import random 
from typing import List
from dataclasses import dataclass 

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
        cov = T.T @ self.cov3D.T @ T

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

    def get_nonopt_bb(self, camera: Camera, thresh:float=3., ):
        bboxsize_cam = self.get_fast_bb(thresh)
        
        bboxsize_ndc = bboxsize_cam# np.clip(bboxsize_cam, -1, 1)

        vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) # this is actually unneccessary... TODO
        bboxsize_cam = np.multiply(vertices, bboxsize_cam) # assuming bboxsize_cam has bounds on x and y for the ellipse, this bounds ul, ur, ll, lr. In principle, that's unneccesary as it uses 8, and not 4, coords (TODO)
        g_pos_ndc = self.get_pos_ndc(camera)
        
        bbox_ndc = np.multiply(vertices, bboxsize_ndc) + g_pos_ndc[:2]
        bbox_ndc = np.hstack((bbox_ndc, np.zeros((vertices.shape[0],2)))) # concatenate bbox with "dummy" zeros
        bbox_ndc[:,2:4] = g_pos_ndc[2:4] # replace the dummy zeros above with z and w from clip position of gaussian. TODO: This is strange, in light of the way we define bboxsize_cam 
        
        return bboxsize_cam, bbox_ndc

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

        vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) # this is actually unneccessary... TODO
        bboxsize_cam = np.multiply(vertices, bboxsize_cam) # assuming bboxsize_cam has bounds on x and y for the ellipse, this bounds ul, ur, ll, lr. In principle, that's unneccesary as it uses 8, and not 4, coords (TODO)
        g_pos_ndc = self.get_pos_ndc(camera)
        
        bbox_ndc = np.multiply(vertices, bboxsize_ndc) + g_pos_ndc[:2]
        bbox_ndc = np.hstack((bbox_ndc, np.zeros((vertices.shape[0],2)))) # concatenate bbox with "dummy" zeros
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

    def get_prec(self, sigma: np.ndarray=None, camera: Camera=None) -> np.ndarray:
        if sigma is None: sigma = self.get_cov2d(camera)
        else: assert sigma.shape == (2, 2)
        det = np.linalg.det(sigma)
        if det < 1e-14:
            warnings.warn(f'{det=} is a bad condition number')
        det_inv = 1.0 / max([det, 1e-14])
        cov = [sigma[0,0], sigma[0,1], sigma[1,1]] # unique elements of covariance matrix
        conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]) 
        return conic 

    def get_exponent(self, x: np.array, camera: Camera=None, mu: np.array=None, sigma: np.ndarray=None) -> float: # TODO: make vectorized version
        if mu is None: 
            mu = self.get_pos_ndc(camera)[:2]
            mu = camera.ndc_to_pixel(mu)
        if sigma is None: 
            sigma = self.get_cov2d()
            sigma_ndc = sigma # / np.array([camera.w, camera.h]) * 2 # TODO: are you sure?
            sigma = camera.ndc_to_pixel(sigma_ndc)
        prec = self.get_prec(sigma)
        prec = np.array([[prec[0], prec[1]], [prec[1], prec[2]]])
        diff = (x - mu[:2]) / np.array([camera.w, camera.h]) # TODO: is this true?
        tmp = (prec @ diff)
        return - diff.T @ tmp / 2

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

