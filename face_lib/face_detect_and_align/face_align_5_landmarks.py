# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/face_lib/face_detect_and_align/face_align_5_landmarks.py
# Compiled at: 2024-03-27 17:14:52
# Size of source mod 2**32: 5949 bytes
import numpy as np, cv2
from cv2box.utils.math import Normalize
from cv2box import CVImage
from .scrfd_insightface import SCRFD
from face_lib.face_detect_and_align.face_align_utils import norm_crop, apply_roi_func
SCRFD_MODEL_PATH = "pretrain_models/face_lib/face_detect/scrfd_onnx/"
MTCNN_MODEL_PATH = "pretrain_models/face_lib/face_detect/mtcnn_weights/"

class FaceDetect5Landmarks:

    def __init__(self, mode='scrfd_500m', tracking=False):
        self.mode = mode
        self.tracking = tracking
        self.dis_list = []
        self.last_bboxes_ = []
        assert self.mode in ('scrfd', 'scrfd_500m', 'mtcnn')
        self.bboxes = self.kpss = self.image = None
        if "scrfd" in self.mode:
            if self.mode == "scrfd_500m":
                scrfd_model_path = SCRFD_MODEL_PATH + "scrfd_500m_bnkps_shape640x640.onnx"
            else:
                scrfd_model_path = SCRFD_MODEL_PATH + "scrfd_10g_bnkps.onnx"
            self.det_model_scrfd = SCRFD(scrfd_model_path)
            self.det_model_scrfd.prepare(ctx_id=0, input_size=(640, 640))

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0, min_bbox_size=None):
        """
        Args:
            image: RGB image path or Numpy array load by cv2
            nms_thresh:
            max_num:
            min_bbox_size:
        Returns:
        """
        self.image = CVImage(image).rgb()
        if self.tracking:
            if len(self.last_bboxes_) == 0:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(image, thresh=nms_thresh, max_num=1, metric="default")
                self.last_bboxes_ = self.bboxes
            else:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(image, thresh=nms_thresh, max_num=0, metric="default")
                self.bboxes, self.kpss = self.tracking_filter()
        else:
            if "scrfd" in self.mode:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces((self.image), thresh=nms_thresh, max_num=max_num,
                  metric="default")
            return (
             self.bboxes, self.kpss)

    def tracking_filter(self):
        for i in range(len(self.bboxes)):
            self.dis_list.append(np.linalg.norm(Normalize(self.bboxes[i]).np_norm() - Normalize(self.last_bboxes_[0]).np_norm()))

        if not self.dis_list:
            return ([], [])
        best_index = np.argmin(np.array(self.dis_list))
        self.dis_list = []
        self.last_bboxes_ = [self.bboxes[best_index]]
        return (
         self.last_bboxes_, [self.kpss[best_index]])

    def bboxes_filter(self, min_bbox_size):
        min_area = np.power(min_bbox_size, 2)
        area_list = (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
        min_index = np.where(area_list < min_area)
        self.bboxes = np.delete((self.bboxes), min_index, axis=0)
        self.kpss = np.delete((self.kpss), min_index, axis=0)

    def get_single_face(self, crop_size, mode='mtcnn_512', apply_roi=False):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface default_95
        Returns: cv2 image
        """
        assert mode in ('default', 'mtcnn_512', 'mtcnn_256', 'arcface_512', 'arcface',
                        'default_95')
        if self.bboxes.shape[0] == 0:
            return (None, None)
        det_score = self.bboxes[(Ellipsis, 4)]
        if self.tracking:
            best_index = np.argmax(np.array(self.dis_list))
            kpss = None
            if self.kpss is not None:
                kpss = self.kpss[best_index]
        else:
            best_index = np.argmax(det_score)
            kpss = None
            if self.kpss is not None:
                kpss = self.kpss[best_index]
        if apply_roi:
            (roi, roi_box, roi_kpss) = apply_roi_func(self.image, self.bboxes[best_index], kpss)
            (align_img, mat_rev) = norm_crop(roi, roi_kpss, crop_size, mode=mode)
            align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
            return (
             align_img, mat_rev, roi_box)
        (align_img, M) = norm_crop((self.image), kpss, crop_size, mode=mode)
        align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
        return (
         align_img, M)

    def get_multi_face(self, crop_size, mode='mtcnn_512'):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface
        Returns:
        """
        if self.bboxes.shape[0] == 0:
            return
        align_img_list = []
        M_list = []
        for i in range(self.bboxes.shape[0]):
            kps = None
            if self.kpss is not None:
                kps = self.kpss[i]
            (align_img, M) = norm_crop((self.image), kps, crop_size, mode=mode)
            align_img_list.append(align_img)
            M_list.append(M)

        return (align_img_list, M_list)

    def draw_face(self):
        for i_ in range(self.bboxes.shape[0]):
            bbox = self.bboxes[i_]
            (x1, y1, x2, y2, score) = bbox.astype(int)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if self.kpss is not None:
                kps = self.kpss[i_]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(self.image, tuple(kp), 1, (0, 0, 255), 2)

            CVImage((self.image), image_format="cv2").show()

