# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/face_lib/face_detect_and_align/face_align_utils.py
# Compiled at: 2024-03-27 17:14:52
# Size of source mod 2**32: 4937 bytes
import cv2, numpy as np
from skimage import transform as trans
src1 = np.array([[51.642, 50.115], [57.617, 49.99], [35.74, 69.007],
 [
  51.157, 89.05], [57.025, 89.702]],
  dtype=(np.float32))
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
 [
  45.177, 86.19], [64.246, 86.758]],
  dtype=(np.float32))
src3 = np.array([[39.73, 51.138], [72.27, 51.138], [56.0, 68.493],
 [
  42.463, 87.01], [69.537, 87.01]],
  dtype=(np.float32))
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
 [
  48.167, 86.758], [67.236, 86.19]],
  dtype=(np.float32))
src5 = np.array([[54.796, 49.99], [60.771, 50.115], [76.673, 69.007],
 [
  55.388, 89.702], [61.257, 89.05]],
  dtype=(np.float32))
multi_src = np.array([src1,src2,src3,src4,src5])
multi_src_map = {112:multi_src,  224:multi_src * 2,  512:multi_src * 4.571428571428571}
arcface_src = np.array([
 [
  38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
 [
  41.5493, 92.3655], [70.7299, 92.2041]],
  dtype=(np.float32))
mtcnn_512 = [
 [
  187.20187, 239.27705],
 [
  324.1236, 238.51973],
 [
  256.09793, 317.14795],
 [
  199.84871, 397.30597],
 [
  313.2362, 396.6788]]
mtcnn_256 = np.array(mtcnn_512) * 0.5
arcface_src_512 = arcface_src * np.array([4.571428571428571, 4.571428571428571])
arcface_src = np.expand_dims(arcface_src, axis=0)

def get_src_modify(srcs, arcface_src):
    srcs += ((arcface_src[2] - srcs[2][2]) * np.array([1, 1.8]))[None]
    return srcs


def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=(np.ones(5)), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    if mode == "arcface":
        assert image_size == 112
        src = arcface_src
    elif mode == "arcface_512":
        src = np.expand_dims(arcface_src_512, axis=0)
    elif mode == "mtcnn_512":
        src = np.expand_dims(mtcnn_512, axis=0)
    elif mode == "mtcnn_256":
        src = np.expand_dims(mtcnn_256, axis=0)
    elif mode == "default_95":
        src = get_src_modify(multi_src, arcface_src[0])
        src_map = {112:(src.copy)(),  224:(src.copy()) * 2,  256:(src.copy() * 256 / 112) * 0.95,  512:(src.copy() * 4.571428571428571) * 0.95}
        src = src_map[image_size]
    else:
        src = multi_src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum(((results - src[i]) ** 2), axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
        return (min_M, min_index)


def norm_crop(img, landmark, crop_size=112, mode='arcface'):
    (mat, pose_index) = estimate_norm(landmark, crop_size, mode)
    warped = cv2.warpAffine(img, mat, (crop_size, crop_size), borderMode=(cv2.BORDER_REPLICATE))
    mat_rev = cv2.invertAffineTransform(mat)
    return (
     warped, mat_rev)


def apply_roi_func(img, box, facial5points):
    box = np.round(np.array(box)).astype(int)[:4]
    roi_pad = 0
    roi_box = np.array([
     max(0, box[0] - roi_pad),
     max(0, box[1] - roi_pad),
     min(img.shape[1], box[2] + roi_pad),
     min(img.shape[0], box[3] + roi_pad)])
    roi = img[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]].copy()
    mrow1 = roi_box[1]
    mcol1 = roi_box[0]
    roi_facial5points = facial5points.copy()
    roi_facial5points[:, 0] -= mcol1
    roi_facial5points[:, 1] -= mrow1
    return (
     roi, roi_box, roi_facial5points)

