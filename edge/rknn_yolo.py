# -*- coding: utf-8 -*-
import cv2
import numpy as np
from rknnlite.api import RKNNLite

class RKNNYOLO:
    def __init__(self, model_path, conf_thres=0.35, iou_thres=0.45, imgsz=640, num_classes=5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    def preprocess(self, img):
        # 直接 resize 到 imgsz x imgsz，保持 uint8
        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        input_data = np.expand_dims(resized.astype(np.uint8), axis=0)  # NHWC
        return input_data

    def postprocess(self, outputs, orig_shape):
        """
        处理 YOLOv8 DFL 输出（多尺度）
        outputs: 列表，每个元素形状 (1, C, H, W)
        orig_shape: (H, W)
        返回: list of (x1, y1, x2, y2, conf)
        """
        # 分离 box (C=64) 和 cls (C=num_classes)
        box_list = []
        cls_list = []
        for out in outputs:
            c = out.shape[1]
            if c == 64:
                box_list.append(out[0])  # (64, H, W)
            elif c == self.num_classes:
                cls_list.append(out[0])  # (num_classes, H, W)

        # 按空间尺寸匹配
        box_dict = {(b.shape[1], b.shape[2]): b for b in box_list}
        cls_dict = {(c.shape[1], c.shape[2]): c for c in cls_list}
        common_shapes = set(box_dict.keys()) & set(cls_dict.keys())

        if not common_shapes:
            print("No matching (H,W) between box and cls features!")
            return []

        dfl_proj = np.arange(16, dtype=np.float32)
        all_boxes = []
        all_scores = []
        all_classes = []

        for (h, w) in common_shapes:
            box_feat = box_dict[(h, w)]  # (64, h, w)
            cls_feat = cls_dict[(h, w)]  # (num_classes, h, w)
            stride = self.imgsz // h

            # 生成网格坐标
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)

            # 展平
            box_flat = box_feat.reshape(64, -1).T  # (N, 64)
            cls_flat = cls_feat.reshape(self.num_classes, -1).T  # (N, num_classes)

            # 置信度：类别最大分数
            max_cls = cls_flat.max(axis=1)
            mask = max_cls >= self.conf_thres
            if not np.any(mask):
                continue

            box_selected = box_flat[mask]          # (N', 64)
            cls_selected = cls_flat[mask]          # (N', num_classes)
            grid_selected = grid[mask]             # (N', 2)
            scores = max_cls[mask]
            class_ids = np.argmax(cls_selected, axis=1)

            # DFL 解码
            b = box_selected.reshape(-1, 4, 16)
            b = b - b.max(axis=2, keepdims=True)
            b = np.exp(b)
            b = b / b.sum(axis=2, keepdims=True)
            dist = (b * dfl_proj).sum(axis=2)      # (N', 4) [ltrb]

            x1 = (grid_selected[:, 0] + 0.5 - dist[:, 0]) * stride
            y1 = (grid_selected[:, 1] + 0.5 - dist[:, 1]) * stride
            x2 = (grid_selected[:, 0] + 0.5 + dist[:, 2]) * stride
            y2 = (grid_selected[:, 1] + 0.5 + dist[:, 3]) * stride

            all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            all_scores.append(scores)
            all_classes.append(class_ids)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)

        # 只保留 people 类 (class_id == 0)
        people_mask = (classes == 0)
        if not np.any(people_mask):
            return []
        boxes = boxes[people_mask]
        scores = scores[people_mask]

        # 缩放回原图尺寸（无填充）
        orig_h, orig_w = orig_shape[:2]
        scale_w = orig_w / self.imgsz
        scale_h = orig_h / self.imgsz
        boxes[:, 0] *= scale_w
        boxes[:, 1] *= scale_h
        boxes[:, 2] *= scale_w
        boxes[:, 3] *= scale_h

        # 裁剪
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        # NMS
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        rects = np.concatenate([boxes[:, 0:2], wh], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.conf_thres, self.iou_thres)
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()
        boxes = boxes[keep]
        scores = scores[keep]

        detections = []
        for box, score in zip(boxes, scores):
            detections.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))
        return detections

    def detect(self, img):
        input_data = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[input_data])
        # outputs is a list of numpy arrays
        detections = self.postprocess(outputs, img.shape)
        return detections

    def release(self):
        self.rknn.release()