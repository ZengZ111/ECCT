# -*- coding: utf-8 -*-
# rknn_tracker.py
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ---------- 默认参数（与 configv3.yaml 一致） ----------
DEFAULT_TRACK_PARAMS = {
    'WINDOW_INFLUENCE': 0.455,
    'PENALTY_K': 0.138,
    'LR': 0.348,
    'EXEMPLAR_SIZE': 127,
    'INSTANCE_SIZE': 255,
    'CONTEXT_AMOUNT': 0.5,
    'OUTPUT_SIZE': 15,
    'STRIDE': 16,
    'CLS_OUT_CHANNELS': 2,
}

# ---------- 辅助函数 ----------
def get_subwindow_np(im, pos, model_sz, original_sz, avg_chans):
    """
    提取图像子窗口（不归一化，保持 uint8）
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                         int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                      int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))

    im_patch = np.expand_dims(im_patch, axis=0)
    return im_patch

def corner2center_np(corner):
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


# ---------- RKNNTracker 类 ----------
class RKNNTracker:
    def __init__(self, backbone_path, backbone_search_path, head_path,
                 params=None, core_mask=None):
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0_1_2  # 使用全部核心

        self.params = params if params else DEFAULT_TRACK_PARAMS.copy()

        # 加载模型（与之前一样）
        self.backbone = RKNNLite()
        self.backbone.load_rknn(backbone_path)
        self.backbone.init_runtime(core_mask=core_mask)

        self.backbone_search = RKNNLite()
        self.backbone_search.load_rknn(backbone_search_path)
        self.backbone_search.init_runtime(core_mask=core_mask)

        self.head = RKNNLite()
        self.head.load_rknn(head_path)
        self.head.init_runtime(core_mask=core_mask)

        # 状态变量
        self.score_size = self.params['OUTPUT_SIZE']
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.cls_out_channels = self.params['CLS_OUT_CHANNELS']
        self.points = self._generate_points(self.params['STRIDE'], self.score_size)

        self.zf = None
        self.center_pos = None
        self.size = None
        self.channel_average = None

    def _generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid(
            [ori + stride * dx for dx in np.arange(0, size)],
            [ori + stride * dy for dy in np.arange(0, size)]
        )
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0] = x.astype(np.float32).flatten()
        points[:, 1] = y.astype(np.float32).flatten()
        return points

    def _run_backbone(self, image_np, is_template=True):
        session = self.backbone if is_template else self.backbone_search
        outputs = session.inference(inputs=[image_np], data_format=['nhwc'])
        return outputs[0]  # NCHW

    def _run_head(self, z_feat, x_feat):
        # 输入 NCHW -> 转 NHWC
        z_nhwc = z_feat.transpose(0, 2, 3, 1)
        x_nhwc = x_feat.transpose(0, 2, 3, 1)
        outputs = self.head.inference(inputs=[z_nhwc, x_nhwc], data_format=['nhwc', 'nhwc'])
        # 输出 NHWC，直接返回（不转回，因为后处理已适配）
        return outputs[0], outputs[1]  # (H,W,C) 格式

    def _convert_bbox(self, delta, point):
        # delta shape: (1, H, W, C) 因为我们的 head 输出 NHWC
        # 需要转换为 (4, N) 并解析
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)
        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center_np(delta)
        return delta

    def _convert_score(self, score):
        # score shape: (1, H, W, C) 因为我们的 head 输出 NHWC
        if self.cls_out_channels == 1:
            score = score.transpose(1, 2, 3, 0).reshape(-1)
            score = 1.0 / (1.0 + np.exp(-score))
        else:
            score = score.transpose(1, 2, 3, 0).reshape(self.cls_out_channels, -1)
            score = score.transpose(1, 0)
            score_max = np.max(score, axis=1, keepdims=True)
            exp_score = np.exp(score - score_max)
            score = exp_score / np.sum(exp_score, axis=1, keepdims=True)
            score = score[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, frame, bbox):
        self.center_pos = np.array([
            bbox[0] + (bbox[2] - 1) / 2,
            bbox[1] + (bbox[3] - 1) / 2
        ])
        self.size = np.array([bbox[2], bbox[3]])

        w_z = self.size[0] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        h_z = self.size[1] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.channel_average = np.mean(frame, axis=(0, 1))

        z_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['EXEMPLAR_SIZE'],
            s_z, self.channel_average
        )
        self.zf = self._run_backbone(z_crop, is_template=True)

    def track(self, frame):
        w_z = self.size[0] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        h_z = self.size[1] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.params['EXEMPLAR_SIZE'] / s_z
        s_x = s_z * (self.params['INSTANCE_SIZE'] / self.params['EXEMPLAR_SIZE'])

        x_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['INSTANCE_SIZE'],
            round(s_x), self.channel_average
        )

        xf = self._run_backbone(x_crop, is_template=False)
        cls, loc = self._run_head(self.zf, xf)

        score = self._convert_score(cls)
        pred_bbox = self._convert_bbox(loc, self.points)

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :]) /
            sz(self.size[0] * scale_z, self.size[1] * scale_z)
        )
        r_c = change(
            (self.size[0] / self.size[1]) /
            (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * self.params['PENALTY_K'])

        pscore = penalty * score
        pscore = pscore * (1 - self.params['WINDOW_INFLUENCE']) + \
                 self.window * self.params['WINDOW_INFLUENCE']

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * self.params['LR']

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, frame.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        final_bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = float(pscore[best_idx])

        return {'bbox': final_bbox, 'score': best_score}

    def release(self):
        if self.backbone:
            self.backbone.release()
        if self.backbone_search:
            self.backbone_search.release()
        if self.head:
            self.head.release()