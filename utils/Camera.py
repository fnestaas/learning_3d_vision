# Adapted from the version at https://github.com/limacv/GaussianSplattingViewer/blob/main/util.py
# TODO: still limitations. E.g. pitch and yaw are not used... 
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
import glm
import ctypes

class Camera:
    def __init__(self, h, w, position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), up: np.array=np.array([0.0, -1.0, 0.0])):
        self.znear = 0.01
        self.zfar = 100
        # self.znear = .9 * max([w, h])
        # self.zfar = 1.1 * max([w, h])
        self.h = h
        self.w = w
        self.fovy = np.pi / 2.0
        self.position = np.array(position)
        self.target = np.array(target)
        self.up = up

    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        """Projection matrix from world to view (camera) space"""
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_projection_matrix(self):
        """view space to clip space"""
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        """
        get the tan of self.fovy/2, and scaled version for x (which is tan(self.fovx/2)). Additionally, return focal, computed as
        focal = self.h / (2 * tan(self.fovy/2)). This is the "depth" (adjacent in sohcahtoa) at which we see height h (the "focal point", I guess)
        (NOT the hyperbolic tan...)
        """
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def get_htanfovxy(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        return [htanx, htany]

    def world_to_cam(self, points):
        view_mat = self.get_view_matrix()

        # if points is 3xN, add a fourth row of ones
        if points.shape[0] == 3:
            # If there is only one point, just add a fourth vallue of 1
            if len(points.shape) == 1:
                points = np.append(points, 1.0)
            else:
                points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

        return np.matmul(view_mat, points)

    def cam_to_world(self, points):
        view_mat = self.get_view_matrix()
        return np.matmul(view_mat.T, points)

    def cam_to_ndc(self, points):
        proj_mat = self.get_projection_matrix()
        points_ndc = proj_mat @ points
        if len(points_ndc.shape) == 1:
            points_ndc = points_ndc / points_ndc[3]
        else:
            points_ndc = points_ndc / points_ndc[3, :]
        return points_ndc

    def ndc_to_cam(self, points):
        proj_mat = self.get_projection_matrix()
        return np.linalg.inv(proj_mat) @ points

    def ndc_to_pixel(self, points_ndc, screen_width=None, screen_height=None):
        """scale linearly from [-1, 1]^2 to [0, width] x [0, height]"""
        # Use camera plane size if screen size not specified
        if screen_width is None:
            screen_width = self.w
        if screen_height is None:
            screen_height = self.h

        width_half = screen_width / 2
        height_half = screen_height / 2

        if len(points_ndc.shape) == 1:
            # It is a single point, so just return the pixel coordinates
            return np.array([(points_ndc[0] + 1) * width_half, (1.0 - points_ndc[1]) * height_half])
        else:
            return np.array([(point[0] * width_half + width_half, - point[1] * height_half + height_half)
                for point in points_ndc])

    def update_resolution(self, height, width):
        self.h = height
        self.w = width
        self.is_intrin_dirty = True
