#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort   # 替换 RKNNLite

# =========================
# 参数解析
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 ONNX model on dataset")
    parser.add_argument('--dataset', required=True, help='Path to dataset directory (contains images and XMLs)')
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--imgsz', type=int, default=640, help='Model input size (square)')
    parser.add_argument('--iou_thres', type=float, default=0.2, help='IoU threshold for matching')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--person_class', type=int, default=0, help='Class ID for person (default 0)')
    # 移除 device_id 参数（RKNN 专用）
    return parser.parse_args()

def compute_ap(detections, gts, iou_thresh=0.5):
    """
    计算单类别的 AP（VOC 2010 11点插值法）
    detections: list of (img_id, conf, x1, y1, x2, y2)
    gts: list of (img_id, x1, y1, x2, y2)
    """
    gt_by_img = {}
    for img_id, x1, y1, x2, y2 in gts:
        gt_by_img.setdefault(img_id, []).append([x1, y1, x2, y2])

    all_detections = sorted(detections, key=lambda x: x[1], reverse=True)

    gt_total = sum(len(v) for v in gt_by_img.values())
    if gt_total == 0:
        return 0.0

    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))

    matched_gt = {img_id: [False] * len(gt_by_img.get(img_id, [])) for img_id in gt_by_img.keys()}

    for idx, (img_id, conf, x1, y1, x2, y2) in enumerate(all_detections):
        gt_boxes = gt_by_img.get(img_id, [])
        if len(gt_boxes) == 0:
            fp[idx] = 1
            continue

        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if matched_gt[img_id][j]:
                continue
            iou = compute_iou([x1, y1, x2, y2], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thresh:
            tp[idx] = 1
            matched_gt[img_id][best_gt_idx] = True
        else:
            fp[idx] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / gt_total if gt_total > 0 else np.zeros_like(tp_cum)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

# =========================
# IoU 计算
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / (areaA + areaB - inter_area)

# =========================
# DFL 后处理（适配多尺度输出）
# =========================
def dfl_decode(box_feat, stride, grid, dfl_proj):
    """
    box_feat: [N, 64]  (N = h*w)
    stride: int
    grid: [N, 2]  (x, y) 网格坐标（模型坐标系）
    dfl_proj: [16] 投影向量
    """
    b = box_feat.reshape(-1, 4, 16)          # [N, 4, 16]
    b = b - b.max(axis=2, keepdims=True)
    b = np.exp(b)
    b = b / b.sum(axis=2, keepdims=True)
    dist = (b * dfl_proj).sum(axis=2)        # [N, 4]  ltrb
    x1 = (grid[:, 0] + 0.5 - dist[:, 0]) * stride
    y1 = (grid[:, 1] + 0.5 - dist[:, 1]) * stride
    x2 = (grid[:, 0] + 0.5 + dist[:, 2]) * stride
    y2 = (grid[:, 1] + 0.5 + dist[:, 3]) * stride
    return np.stack([x1, y1, x2, y2], axis=-1)

def post_process_dfl(outputs, imgsz, conf_thres, nms_thres=0.45, num_classes=5):
    """
    自动识别 box 和 cls 特征，按空间尺寸配对。
    outputs: 列表，每个元素 shape [1, C, H, W]
    """
    box_list = []
    cls_list = []
    for idx, out in enumerate(outputs):
        c = out.shape[1]
        if c == 64:
            box_list.append((out.shape[2], out.shape[3], out))
        elif c == num_classes:
            cls_list.append((out.shape[2], out.shape[3], out))
        else:
            pass

    box_dict = {(h, w): tensor for h, w, tensor in box_list}
    cls_dict = {(h, w): tensor for h, w, tensor in cls_list}
    common_shapes = set(box_dict.keys()) & set(cls_dict.keys())
    if not common_shapes:
        print("No matching (H,W) between box and cls features!")
        return []

    dfl_proj = np.arange(16, dtype=np.float32)
    all_boxes, all_scores, all_classes = [], [], []

    for (h, w) in common_shapes:
        box_feat = box_dict[(h, w)]   # [1,64,h,w]
        cls_feat = cls_dict[(h, w)]   # [1,num_classes,h,w]
        stride = imgsz // h

        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)

        box_flat = box_feat.reshape(64, -1).T          # [h*w, 64]
        cls_flat = cls_feat.reshape(num_classes, -1).T # [h*w, num_classes]
        cls_max = cls_flat.max(axis=1)
        mask = cls_max >= conf_thres
        if not np.any(mask):
            continue

        box_selected = box_flat[mask]
        cls_selected = cls_flat[mask]
        grid_selected = grid[mask]
        scores = cls_max[mask]
        class_ids = np.argmax(cls_selected, axis=1)

        b = box_selected.reshape(-1, 4, 16)
        b = b - b.max(axis=2, keepdims=True)
        b = np.exp(b)
        b = b / b.sum(axis=2, keepdims=True)
        dist = (b * dfl_proj).sum(axis=2)

        x1 = (grid_selected[:, 0] + 0.5 - dist[:, 0]) * stride
        y1 = (grid_selected[:, 1] + 0.5 - dist[:, 1]) * stride
        x2 = (grid_selected[:, 0] + 0.5 + dist[:, 2]) * stride
        y2 = (grid_selected[:, 1] + 0.5 + dist[:, 3]) * stride
        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(class_ids)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    people_mask = (classes == 0)
    if not np.any(people_mask):
        return []
    boxes = boxes[people_mask]
    scores = scores[people_mask]
    classes = classes[people_mask]

    wh = boxes[:, 2:4] - boxes[:, 0:2]
    rects = np.concatenate([boxes[:, 0:2], wh], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf_thres, nms_thres)
    if len(keep) == 0:
        return []
    keep = np.array(keep).flatten()
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        detections.append([box[0], box[1], box[2], box[3], float(score), int(cls_id)])
    return detections

# =========================
# ONNX 推理（替换原 rknn_inference）
# =========================
def onnx_inference(sess, img_orig, imgsz=640, conf_thres=0.5, nms_thres=0.45):
    t_pre_start = time.time()
    img_h, img_w = img_orig.shape[:2]

    # 预处理（与 RKNN 版本保持一致，直接 resize 为 uint8）
    img_resized = cv2.resize(img_orig, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 注意：ONNX 模型可能期望 float 归一化，若模型需要请取消下一行注释并注释掉 uint8 版本
    input_data = np.expand_dims((img_rgb.astype(np.float32) / 255.0).transpose(2,0,1), axis=0)
    #input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)  # 保持与 RKNN 一致
    t_pre_end = time.time()
    preprocess_time = (t_pre_end - t_pre_start) * 1000

    # 推理（ONNX Runtime）
    t_inf_start = time.time()
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_data})   # outputs 为列表，每个元素 [1,C,H,W]
    t_inf_end = time.time()
    inference_time = (t_inf_end - t_inf_start) * 1000

    # 后处理（DFL）
    t_post_start = time.time()
    detections = post_process_dfl(outputs, imgsz, conf_thres, nms_thres, num_classes=5)
    t_post_end = time.time()
    postprocess_time = (t_post_end - t_post_start) * 1000

    # 缩放至原图尺寸
    scale_w = img_w / imgsz
    scale_h = img_h / imgsz
    scaled_detections = []
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        x1 = int(np.clip(x1 * scale_w, 0, img_w - 1))
        y1 = int(np.clip(y1 * scale_h, 0, img_h - 1))
        x2 = int(np.clip(x2 * scale_w, 0, img_w - 1))
        y2 = int(np.clip(y2 * scale_h, 0, img_h - 1))
        scaled_detections.append([x1, y1, x2, y2, score, cls_id])

    return scaled_detections, preprocess_time, inference_time, postprocess_time

# =========================
# 主函数
# =========================
def main():
    args = parse_args()

    # ======== 1. 加载 ONNX 模型 ========
    print("Loading ONNX model...")
    # 使用 CPU 执行，若需多线程可在此设置 SessionOptions
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    print("Model loaded successfully.")

    # ======== 2. 获取图片列表 ========
    dataset_dir = args.dataset
    jpg_files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith('.jpg')])
    print(f"Found {len(jpg_files)} images")

    # ======== 3. 评估 ========
    conf_thres = args.conf
    print(f"Confidence Threshold = {conf_thres:.2f}")
    print(f"IoU Threshold = {args.iou_thres:.2f}")

    gt_total = 0
    tp_total = 0
    pred_total = 0
    class_counter = Counter()

    total_preprocess = 0.0
    total_inference = 0.0
    total_postprocess = 0.0
    processed_count = 0
    all_detections = []
    all_gts = []

    for idx, img_name in enumerate(jpg_files):
        img_path = os.path.join(dataset_dir, img_name)
        xml_path = img_path.replace('.jpg', '.xml')
        if not os.path.exists(xml_path):
            continue

        # ---- 读取 GT ----
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            gt_boxes.append([xmin, ymin, xmax, ymax])
            all_gts.append((img_name, xmin, ymin, xmax, ymax))
        gt_count = len(gt_boxes)

        # ---- 读取原图 ----
        img = cv2.imread(img_path)
        if img is None:
            continue

        # ---- ONNX 推理 ----
        detections, preprocess_time, inference_time, postprocess_time = onnx_inference(
            sess, img,
            imgsz=args.imgsz,
            conf_thres=conf_thres
        )
        total_preprocess += preprocess_time
        total_inference += inference_time
        total_postprocess += postprocess_time
        processed_count += 1

        pred_boxes = []
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            class_counter[cls_id] += 1
            pred_boxes.append([x1, y1, x2, y2])
            all_detections.append((img_name, score, x1, y1, x2, y2))

        pred_count = len(pred_boxes)

        # ---- IoU 匹配 ----
        matched_preds = set()
        tp = 0
        for gt_box in gt_boxes:
            best_iou = 0
            best_idx = -1
            for p_idx, pred_box in enumerate(pred_boxes):
                if p_idx in matched_preds:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = p_idx
            if best_iou >= args.iou_thres:
                tp += 1
                matched_preds.add(best_idx)

        gt_total += gt_count
        tp_total += tp
        pred_total += pred_count

        if idx % 50 == 0:
            img_recall = tp / gt_count if gt_count > 0 else 0
            print(f"[{idx}/{len(jpg_files)}] {img_name} GT={gt_count} PRED={pred_count} TP={tp} Recall={img_recall:.3f}")

    # ---- 计算指标 ----
    recall = tp_total / gt_total if gt_total > 0 else 0
    precision = tp_total / pred_total if pred_total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fn_total = gt_total - tp_total

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"GT Total         : {gt_total}")
    print(f"Prediction Total : {pred_total}")
    print(f"TP Total         : {tp_total}")
    print(f"FN Total         : {fn_total}")
    print(f"Recall           : {recall:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    ap = compute_ap(all_detections, all_gts, iou_thresh=0.5)
    print(f"mAP@0.5 (single class) : {ap:.4f}")

    print("\n" + "=" * 50)
    print("TIME STATISTICS")
    print("=" * 50)
    avg_pre = total_preprocess / processed_count if processed_count > 0 else 0
    avg_inf = total_inference / processed_count if processed_count > 0 else 0
    avg_post = total_postprocess / processed_count if processed_count > 0 else 0
    avg_total = avg_pre + avg_inf + avg_post
    fps = 1000.0 / avg_total if avg_total > 0 else 0
    print(f"Preprocess/Image : {avg_pre:.3f} ms")
    print(f"Inference/Image  : {avg_inf:.3f} ms")
    print(f"Postprocess/Image: {avg_post:.3f} ms")
    print(f"Total/Image      : {avg_total:.3f} ms")
    print(f"FPS              : {fps:.2f}")

if __name__ == '__main__':
    main()