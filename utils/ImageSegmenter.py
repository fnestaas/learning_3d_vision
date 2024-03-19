### Old code, might revisit

# from utils.Camera import Camera 
# from utils.Primitives import PrimitiveSet, PrimitiveSubset
# from dataclasses import dataclass
# import numpy as np 
# from typing import List, Tuple, Dict 
# import random 

# @dataclass 
# class SubImage:
#     """This is a utility class for handling SubImages, i.e. a rectangular part of an Image"""
#     image: np.ndarray 
#     ul: tuple # upper left of sub-image
#     lr: tuple # lower right of sub-image
#     camera: Camera 
#     def __init__(
#         self, 
#         image: np.ndarray, 
#         ul: Tuple[float, float], 
#         lr: Tuple[float, float], 
#         camera: Camera
#     ):
#         self.image = image 
#         self.ul = ul 
#         self.lr = lr
#         self.camera = camera 
#         self.ymin, self.xmin = ul 
#         self.ymax, self.xmax = lr 
#         self.bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
    
#     def __getitem__(self, key):
#         # if isinstance(key, tuple):
#         #     key = tuple(np.s_[k] if isinstance(k, slice) else k for i, k in enumerate(key))
#         if isinstance(key, tuple) and any([isinstance(k, slice) for k in key]): 
#             ul = (0 if key[0].start is None else key[0].start, 0 if key[1].start is None else key[1].start)
#             lr = (self.image.shape[0] if key[0].stop is None else key[0].stop,  self.image.shape[1] if key[1].stop is None else key[1].stop)
#             key = tuple(np.s_[k] if isinstance(k, slice) else k for i, k in enumerate(key))
#             return SubImage(self.image[key], ul, lr, self.camera)
#         return self.image[key]
    
#     @property
#     def shape(self): return self.image.shape

# class ImageSegmenter:
#     """This class will find an optimal way to cut an image, minimizing the maximum number of primitives assigned to one side of the image."""
#     pass 

# class IterativeImageSegmenter:
#     """This class will iteratively use ImageSegmenters to segment the image into multiple parts, which have a "balanced" number of primitives assigned to them"""
#     def __init__(
#         self, 
#         subset: PrimitiveSet, 
#         img: SubImage,
#         camera: Camera,
#         thresh: float=3.,
        
#     ):
#         self.subset = subset 
#         self.img = img 
#         self.thresh = thresh
#         self.camera = camera
#         self.cuts = [
#             {'level': 0, 'responsibility': list(range(len(subset))), 'corners': (img.ul, img.lr)}
#         ] # list of dicts with subimage coordinates, number of primitives belonging to this segment, and the "cut level" - how many larger sub-images is each sub image part of

#     def cut(self, idx: int):
#         """add another two subimages to self.cuts by finding an optimal cut of self.cuts[idx]"""
#         item = self.cuts[idx]
#         ul, lr = item['corners']
#         subimage = self.img[ul[0]:lr[0], ul[1]: lr[1]]
#         ss = [p for i, p in enumerate(self.subset) if i in item['responsibility']]
#         bx, by = cut_image(subimage, ss, self.camera, thresh=self.thresh)
#         if bx is not None:
#             self.cuts.extend(
#                 [
#                     {'level': item['level']+1, 'responsibility': self.get_ids(subimage[:, bx:]), 'corners': ((ul[0], bx), lr)}, # TODO: responsibility, and verify corners
#                     {'level': item['level']+1, 'responsibility': self.get_ids(subimage[:, :bx]), 'corners': (ul, (lr[0], bx))}
#                 ]
#             )
#         else: 
#             self.cuts.extend(
#                 [
#                     {'level': item['level']+1, 'responsibility': self.get_ids(subimage[by:, :]), 'corners': ((by, ul[1]), lr)}, # TODO: responsibility, and verify corners
#                     {'level': item['level']+1, 'responsibility': self.get_ids(subimage[:by, :]), 'corners': (ul, (by, lr[1]))}
#                 ]
#             )
#         return bx, by
    
#     def get_ids(self, subimage: SubImage):
#         return [i for i, p in enumerate(self.subset) if is_member_bb(subimage, p, self.thresh, camera=self.camera)]

# def cut_image(subimage: SubImage, subset: PrimitiveSubset, camera:Camera, tol: float=0.02, thresh:float=3) -> Tuple[float, float]:
#     """Utility function that takes indices defined in gaussians and returns a tuple (x, y), where one is None, that optimally divides the picture"""
#     # binary search to optimize the objective
#     minmax_x = np.inf 
#     xmin, xmax, ymin, ymax = subimage.bounds
#     # search x 
#     lower, upper = xmin, xmax
#     if tol < 1: tol = tol * max(subimage.shape)
#     l0 = len(subset)
#     while upper - lower > tol:
#         assert l0 == len(subset), f'{subset=}, {upper=}, {lower=}'
#         middle = int((lower + upper) / 2)
#         left_count = get_responsibility(subimage[:, :middle], subset, thresh, camera)
#         right_count = get_responsibility(subimage[:, middle:], subset, thresh, camera)
#         if max([left_count, right_count]) < minmax_x: 
#             minmax_x = max([left_count, right_count]) 
#             x_cut = middle 
#         if left_count < right_count: 
#             lower = middle
#         else: 
#             upper = middle 
#     # search y
#     lower, upper = ymin, ymax
#     minmax_y = np.inf 
#     while upper - lower > tol:
#         assert l0 == len(subset), f'{subset=}, {upper=}, {lower=}'
#         middle = int((lower + upper) / 2)
#         left_count = get_responsibility(subimage[:middle, :], subset, thresh, camera)
#         right_count = get_responsibility(subimage[middle:, :], subset, thresh, camera)
#         if max([left_count, right_count]) < minmax_y: 
#             minmax_y = max([left_count, right_count]) 
#             y_cut = middle 
#         if left_count < right_count: lower = middle
#         else: upper = middle
#     print(f'{minmax_x=}, {minmax_y=}')
#     if minmax_x > minmax_y: return (None, y_cut)
#     return (x_cut, None)

# def get_responsibility(subimage: SubImage, subset: PrimitiveSubset, thresh: float, camera:Camera) -> int:
#     """Compute the number of primitives in subset potentially assigned to subimage."""
#     ss = [primitive for primitive in subset if is_member_bb(subimage, primitive, thresh, camera=camera)]
#     # sss =[p for p in subset if not is_member_bb(subimage, p, thresh, camera=camera)]
#     # print(subimage.bounds)
#     # plot_conics_and_bbs(ss, camera, 'blue')
#     # plot_conics_and_bbs(sss, camera, 'red')
#     # plt.xlim([0, camera.w])
#     # plt.ylim([0, camera.h])
#     # plt.grid(True)
#     # plt.show()
#     return len(ss)

# def is_member_bb(subimage: SubImage, primitive: Gaussian, thresh: float, mu: np.array=None, sigma: np.ndarray=None, camera: Camera=None) -> bool:
#     # bbox_ndc = primitive.get_bb_ndc(camera, thresh, optimal=True)[0]
#     bbox_ndc = primitive.get_conic_and_bb(camera, thresh, optimal=True)[-1]
#     bbox_screen = camera.ndc_to_pixel(bbox_ndc)
#     xmin, xmax, ymin, ymax = subimage.bounds
#     ul = bbox_screen[0,:2]
#     lr = bbox_screen[2,:2]
#     intersect_x = xmax >= ul[0] >= xmin or xmax >= lr[0] >= xmin or lr[0] >= xmax >= xmin >=ul[0] # bounds lie within screen (first two), or screen lies within bounds (last)
#     intersect_y = ymax >= lr[1] >= ymin or ymax >= ul[1] >= ymin or lr[1] <= ymin <= ymax <=ul[1]
#     return intersect_x and intersect_y

# # def is_member(subimage: SubImage, primitive: Gaussian, thresh: float, mu: np.array=None, sigma: np.ndarray=None, camera: Camera=None) -> bool:
# #     """Does the primitive potentially belong in subimage?
# #     It seems that if sigma has big eigenvalues, we run into trouble. But let's just use the optimal bounding boxes we computed lol
# #     """
# #     def get_max_line(x: float, y: float):
# #         """for a fixed value of x or y, maximize the primitive on the specified line. Return the maximum value"""
# #         assert x is None or y is None and not (x is None and y is None)
# #         if x is None: 
# #             x = sigma[1, 0] / sigma[1, 1] * y 
# #         else:
# #             y = sigma[1, 0] / sigma[0, 0] * x 
# #         return primitive.get_exponent(np.array([x, y]), camera, mu, sigma)
    
# #     # if the mean is in the image, we assume that it's bright enough to be counted
# #     camera = subimage.camera
# #     if mu is None: 
# #         mu = primitive.get_pos_ndc(camera)[:2]
# #         mu = camera.ndc_to_pixel(mu) # translate to pixel coords
        
# #     xmin, xmax, ymin, ymax = subimage.bounds
# #     if xmin <= mu[0] <= xmax and ymin <= mu[1] <= ymax: return True

# #     # find the maximum value of the primitive on subimage
# #     if sigma is None: 
# #         sigma = primitive.get_cov2d(camera)
# #         # sigma_ndc = sigma / np.array([camera.w, camera.h]) * 2
# #         # sigma = camera.ndc_to_pixel(sigma_ndc)
# #     # if mu[0]<-1: print(mu) 
# #     res = max([
# #         get_max_line(xmin, None),
# #         get_max_line(xmax, None),
# #         get_max_line(None, ymin),
# #         get_max_line(None, ymax),
# #     ])
# #     return res >= -thresh
    