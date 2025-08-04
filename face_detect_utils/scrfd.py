# File Name: /data/zhangmaolin/code/face2face_train/face_detect_utils/scrfd.py
# Python 3.9

from __future__ import division
import datetime
import numpy as np
import onnxruntime
import os
import os.path as osp
import cv2
import sys


def softmax(z):
    """
    Numerically stable softmax function.
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # broadcast
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # broadcast
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        # Note: The original code might have used a tensor library with .clamp().
        # np.clip is the functional equivalent for numpy arrays.
        x1 = np.clip(x1, a_min=0, a_max=max_shape[1])
        y1 = np.clip(y1, a_min=0, a_max=max_shape[0])
        x2 = np.clip(x2, a_min=0, a_max=max_shape[1])
        y2 = np.clip(y2, a_min=0, a_max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """
    Decode distance prediction to keypoints.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to N
            keypoints (N*2).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        # Note: The original bytecode for px calculation was `points[:, i % 2]`,
        # which is likely a bug and should be `points[:, 0]`.
        # Reconstructing with the more logical implementation.
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, a_min=0, a_max=max_shape[1])
            py = np.clip(py, a_min=0, a_max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, session=None, cpu=False):
        self.model_file = model_file

        # --- 【请务必添加下面的详细调试代码】 ---
        path_exists = osp.exists(self.model_file)

        if not path_exists:
            # 打印出路径的详细信息，帮助我们诊断
            print("--- LINUX SCRFD DEBUG INFO ---", flush=True)
            print(f"FATAL: Path assertion failed. File does not exist.", flush=True)
            print(f"Path being checked: {self.model_file}", flush=True)
            # 使用 repr() 可以暴露路径字符串中的隐藏字符
            print(f"Path representation (repr): {repr(self.model_file)}", flush=True)

            # 检查上一级目录是否存在，帮助定位问题
            parent_dir = osp.dirname(self.model_file)
            print(f"Parent directory: {parent_dir}", flush=True)
            print(f"Parent directory exists? {osp.exists(parent_dir)}", flush=True)

            # 尝试列出父目录中的内容
            if osp.exists(parent_dir):
                try:
                    print(f"Contents of parent directory '{parent_dir}':", flush=True)
                    dir_contents = os.listdir(parent_dir)
                    if not dir_contents:
                        print("  (Directory is empty)", flush=True)
                    else:
                        for item in dir_contents:
                            print(f"  - {item}", flush=True)
                except Exception as e:
                    print(f"  Could not list directory contents: {e}", flush=True)

            print("--- END LINUX SCRFD DEBUG INFO ---", flush=True)

        assert path_exists, f"Model file not found at path: {self.model_file}"
        # --- 调试代码结束 ---

        self.session = session
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            if cpu:
                providers = ['CPUExecutionProvider']
            else:
                print('检测人脸使用GPU')
                providers = ['CUDAExecutionProvider']
            self.session = onnxruntime.InferenceSession(self.model_file, None, providers=providers)

        self.center_cache = {}
        self.nms_thresh = 0.35  # Default value, can be overridden in prepare()
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])

        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True

        output_names = [o.name for o in outputs]

        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1

        # Determine model architecture from number of outputs
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh

        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height, input_width = blob.shape[2], blob.shape[3]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.7, input_size=None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = (img.shape[0] // 2, img.shape[1] // 2)

            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0

            bindex = np.argsort(values)[::-1]
            bindex = bindex[:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def nms(self, dets):
        nms_thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        return keep


# The following are factory functions, likely from insightface.model_zoo
def get_scrfd(name, download=False, root='~/.insightface/models', **kwargs):
    if not download:
        assert osp.exists(name)
        return SCRFD(name)
    else:
        # This part requires the insightface.model_zoo, mocking the logic
        try:
            from insightface.model_zoo.model_store import get_model_file
        except ImportError:
            raise ImportError("Please install insightface to use the download feature.")

        _file = get_model_file(f"scrfd_{name}", root=root)
        return SCRFD(_file)


def scrfd_2p5gkps(**kwargs):
    return get_scrfd('2p5gkps', download=True, **kwargs)


if __name__ == '__main__':
    import glob

    # Initialize detector
    detector = SCRFD(
        model_file='/workspace/codes/SimSwapAPI_mini/pretrain_models/insightface_func/scrfd_10g_bnkps.onnx')
    detector.prepare(-1)  # -1 for CPU

    img_paths = ['/workspace/codes/insightface-master/detection/scrfd/tests/data/t1.jpg']

    for img_path in img_paths:
        img = cv2.imread(img_path)

        # Timing the detection
        for _ in range(1):
            ta = datetime.datetime.now()
            bboxes, kpss = detector.detect(img, thresh=0.5, input_size=(640, 640))
            tb = datetime.datetime.now()
            print('all cost:', (tb - ta).total_seconds() * 1000)

        print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)

        # Draw results
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(np.int32)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int32)
                    cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)

        filename = img_path.split('/')[-1]
        print('output:', filename)
        cv2.imwrite(f'./{filename}', img)