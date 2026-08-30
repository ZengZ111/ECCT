#!/usr/bin/env python3
# rknn_eval.py (集成卡尔曼滤波与异常检测) - 支持多模式
import os
import sys
import time
import cv2
import numpy as np
import argparse
from loguru import logger
from pathlib import Path

from rknn_yolo import RKNNYOLO
from rknn_tracker import RKNNTracker
from rknn_client import VLMClient
from kalman_tracker import KalmanBoxTracker

# ================== 辅助函数 ==================
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def normalize_bbox(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return [float(x1), float(y1), float(x2), float(y2)]

def select_best_candidate(candidates, last_bbox, img_shape):
    if not candidates:
        return None
    H, W = img_shape[:2]
    if last_bbox is None:
        return max(candidates, key=lambda x: x[4])[:4]
    best = None
    best_cost = float('inf')
    for (x1, y1, x2, y2, conf) in candidates:
        iou = compute_iou([x1, y1, x2, y2], last_bbox)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        lcx = (last_bbox[0] + last_bbox[2]) / 2
        lcy = (last_bbox[1] + last_bbox[3]) / 2
        dist = np.hypot(cx - lcx, cy - lcy) / max(H, W)
        cost = 0.4 * (1 - iou) + 0.3 * dist + 0.3 * (1 - conf)
        if cost < best_cost:
            best_cost = cost
            best = (x1, y1, x2, y2)
    return best

def read_groundtruth(seq_dir):
    gt_path = os.path.join(seq_dir, "groundtruth.txt")
    if not os.path.exists(gt_path):
        logger.warning(f"groundtruth.txt not found in {seq_dir}")
        return []
    boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split(',')))
            if len(parts) == 4:
                x, y, w, h = parts
                boxes.append([x, y, x + w, y + h])
            elif len(parts) == 8:
                xs = parts[0::2]
                ys = parts[1::2]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                boxes.append([x1, y1, x2, y2])
            else:
                logger.warning(f"Unknown GT format in {gt_path}")
    return boxes

def read_language_prompt(seq_dir):
    lang_path = os.path.join(seq_dir, "language.txt")
    if not os.path.exists(lang_path):
        return "object"
    with open(lang_path, 'r') as f:
        lines = f.readlines()
        if lines:
            return lines[0].strip()
    return "object"

# ================== 单序列评估类 ==================
class SingleSequenceEvaluator:
    def __init__(self, seq_path, text_prompt, pc_ip,
                 yolo_model_path, tracker_model_paths,
                 score_threshold=0.3, overscan=50,
                 max_frames=None,
                 kalman_factor=2.5, anomaly_frames=2,
                 mode='ours', fallback_category=None, disable_fallback=False,
                 save_video=False, video_output_dir="/workspace"):
        self.seq_path = seq_path
        self.text_prompt = text_prompt
        self.score_threshold = score_threshold
        self.overscan = overscan
        self.max_frames = max_frames
        self.kalman_factor = kalman_factor
        self.anomaly_frames = anomaly_frames
        self.mode = mode
        self.fallback_category = fallback_category
        self.disable_fallback = disable_fallback

        # 根据模式设置标志
        self.cloudtrack_mode = (mode == 'cloudtrack')
        self.edge_only_mode = (mode == 'edge_only')
        self.cloud_only_mode = (mode == 'cloud_only')
        self.no_reid_mode = (mode == 'no_reid')
        self.no_anomaly_mode = (mode == 'no_anomaly')
        self.no_score_mode = (mode == 'no_score')

        # 加载模型
        self.yolo = RKNNYOLO(yolo_model_path, conf_thres=0.3, iou_thres=0.45)
        self.tracker = RKNNTracker(*tracker_model_paths)

        # 连接VLM网关（纯端侧模式也会连接，但不会调用）
        self.vlm_client = VLMClient(pc_ip)   # 只传 IP，端口默认 9999
        self.vlm_client.connect()

        # 状态
        self.current_bbox = None
        self.last_success_bbox = None
        self.tracker_initialized = False
        self.iou_list = []
        self.pred_boxes = []
        self.re_detection_count = 0
        self.lost_count = 0
        self.actual_frames = 0
        self.start_time = None
        self.total_upload_bytes = 0
        self.total_upload_crops = 0
        self.upload_frames = 0
        self.yolo_time_total = 0.0
        self.yolo_call_count = 0
        self.tracker_time_total = 0.0
        self.tracker_call_count = 0
        self.total_frame_time = 0.0
        self.kalman = None
        self.anomaly_counter = 0

        self.vlm_call_count = 0
        self.total_vlm_time = 0.0

        self.network_fail_count = 0          # 连续失败次数
        self.network_available = True        # 当前网络可用标志
        self.max_failures = 1                # 判定不可用的阈值
        self.probe_counter = 0
        self.probe_interval = 30  # 每 30 帧探测一次

        self.degradation_count = 0    # 降级触发次数
        self.recovery_count = 0       # 网络恢复次数

        self.gt_boxes = read_groundtruth(seq_path)
        color_dir = os.path.join(seq_path, "color")
        self.image_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.total_available_frames = len(self.image_files)
        if self.total_available_frames != len(self.gt_boxes):
            logger.warning(f"GT length ({len(self.gt_boxes)}) != frames ({self.total_available_frames}) in {seq_path}")

        self.save_video = save_video
        self.video_output_dir = video_output_dir
        if self.save_video:
            os.makedirs(video_output_dir, exist_ok=True)
            video_path = os.path.join(video_output_dir, f"{Path(seq_path).name}.avi")
            self.video_writer = None
            self.video_path = video_path
        else:
            self.video_writer = None

    def run(self):
        self.start_time = time.time()
        total_available = len(self.image_files)
        if self.max_frames is not None:
            actual_total = min(total_available, self.max_frames)
        else:
            actual_total = total_available

        for idx, img_file in enumerate(self.image_files):
            frame_start = time.time()
            if idx >= actual_total:
                break
            img_path = os.path.join(self.seq_path, "color", img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                logger.error(f"Cannot read {img_path}")
                continue
            h, w = frame.shape[:2]

            if idx < len(self.gt_boxes):
                gt = self.gt_boxes[idx]
            else:
                gt = None

            # ---------- 核心处理 ----------
            if idx == 0:
                self._handle_first_frame(frame)
            else:
                if self.tracker_initialized and not self.cloud_only_mode:
                    # 正常跟踪
                    start_tracker = time.time()
                    result = self.tracker.track(frame)
                    self.tracker_time_total += time.time() - start_tracker
                    self.tracker_call_count += 1
                    pred_xywh = result['bbox']
                    track_score = result['score']
                    pred_xyxy = xywh_to_xyxy(pred_xywh)
                    pred_xyxy = normalize_bbox(pred_xyxy, h, w)

                    # 卡尔曼
                    is_anomaly = False
                    if self.kalman is not None:
                        self.kalman.predict()
                        is_anomaly = self.kalman.is_anomaly(pred_xyxy, threshold_factor=self.kalman_factor)
                        _, _ = self.kalman.update(pred_xyxy)

                    # 决策
                    use_score = not self.no_score_mode
                    if self.cloudtrack_mode:
                        use_anomaly = False
                    else:
                        use_anomaly = not self.no_anomaly_mode
                    trigger = False
                    if use_score and track_score < self.score_threshold:
                        trigger = True
                    if use_anomaly and is_anomaly:
                        self.anomaly_counter += 1
                        if self.anomaly_counter >= self.anomaly_frames:
                            trigger = True
                    else:
                        if not is_anomaly:
                            self.anomaly_counter = 0

                    if trigger:
                        self.lost_count += 1
                        self._handle_re_detection(frame)
                    else:
                        self.current_bbox = pred_xyxy
                        self.last_success_bbox = self.current_bbox
                        if not is_anomaly:
                            self.anomaly_counter = 0
                else:
                    # 跟踪器未初始化 或 cloud_only模式每帧都调用重检
                    if self.cloud_only_mode:
                        self._handle_re_detection(frame)
                    else:
                        self._handle_re_detection(frame)

            gt_valid = False
            if gt is not None:
                gt_valid = not (np.isnan(gt).any() or (gt[0] == 1 and gt[1] == 1 and gt[2] == 0 and gt[3] == 0))
                if gt_valid:
                    if self.current_bbox is not None:
                        iou = compute_iou(self.current_bbox, gt)
                        self.iou_list.append(iou)
                        self.pred_boxes.append(self.current_bbox)
                    else:
                        self.iou_list.append(0.0)
                        self.pred_boxes.append(None)

            # 保存视频（略，保持原有逻辑不变）
            if self.save_video and frame is not None:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    self.video_writer = cv2.VideoWriter(
                        self.video_path,
                        cv2.VideoWriter_fourcc(*'XVID'),
                        30,
                        (w, h)
                    )
                vis_frame = frame.copy()
                if gt is not None and gt_valid:
                    gt_int = [int(c) for c in gt]
                    cv2.rectangle(vis_frame, (gt_int[0], gt_int[1]), (gt_int[2], gt_int[3]), (0, 255, 0), 2)
                    cv2.putText(vis_frame, "GT", (gt_int[0], gt_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                if self.current_bbox is not None:
                    track_int = [int(c) for c in self.current_bbox]
                    cv2.rectangle(vis_frame, (track_int[0], track_int[1]), (track_int[2], track_int[3]), (0, 0, 255), 2)
                    cv2.putText(vis_frame, "Track", (track_int[0], track_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                iou_text = "IoU: N/A"
                if gt is not None and gt_valid and self.current_bbox is not None:
                    iou_val = compute_iou(self.current_bbox, gt)
                    iou_text = f"IoU: {iou_val:.3f}"
                cv2.putText(vis_frame, f"Frame: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(vis_frame, iou_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                self.video_writer.write(vis_frame)

            self.actual_frames += 1
            self.total_frame_time += time.time() - frame_start
            if not self.network_available:
                self.probe_counter += 1
                if self.probe_counter >= self.probe_interval:
                    self.probe_counter = 0
                    self._probe_network(frame)
            if idx % 50 == 0:
                logger.info(f"  Frame {idx+1}/{actual_total} done.")

        elapsed = time.time() - self.start_time
        miou = np.mean(self.iou_list) if self.iou_list else 0.0
        fps = self.actual_frames / elapsed if elapsed > 0 else 0
        success_rate = sum(1 for i in self.iou_list if i > 0.5) / len(self.iou_list) if self.iou_list else 0

        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video saved to {self.video_path}")

        stats = {
            'seq_name': os.path.basename(self.seq_path),
            'miou': miou,
            'fps': fps,
            'success_rate': success_rate,
            'total_frames': self.actual_frames,
            'iou_count': len(self.iou_list),
            're_detections': self.re_detection_count,
            'lost_count': self.lost_count,
            'elapsed': elapsed,
            'vlm_calls': self.vlm_call_count,
            'vlm_time': self.total_vlm_time,
            'total_upload_mb': self.total_upload_bytes / (1024 * 1024),
            'avg_upload_per_frame_kb': (self.total_upload_bytes / self.actual_frames) / 1024 if self.actual_frames else 0,
            'avg_crops_per_frame': self.total_upload_crops / self.upload_frames if self.upload_frames else 0,
            'avg_yolo_ms': (self.yolo_time_total / self.yolo_call_count * 1000) if self.yolo_call_count else 0,
            'avg_tracker_ms': (self.tracker_time_total / self.tracker_call_count * 1000) if self.tracker_call_count else 0,
            'avg_vlm_ms': (self.total_vlm_time / self.vlm_call_count * 1000) if self.vlm_call_count else 0,
            'avg_frame_ms': (self.total_frame_time / self.actual_frames * 1000) if self.actual_frames else 0,
            'degradation_count': self.degradation_count,
            'recovery_count': self.recovery_count,
        }
        return stats

    # ---------- 首帧 ----------
    def _handle_first_frame(self, frame):
        h, w = frame.shape[:2]
        start_yolo = time.time()
        dets = self.yolo.detect(frame)
        self.yolo_time_total += time.time() - start_yolo
        self.yolo_call_count += 1

        if self.cloud_only_mode:
            short_cat = self.fallback_category if self.fallback_category else "person"
            start_vlm = time.time()
            boxes, _, sent_bytes = self.vlm_client.detect_and_filter(
                frame, short_cat, self.text_prompt, self.overscan
            )
            vlm_elapsed = time.time() - start_vlm
            self.vlm_call_count += 1
            self.total_vlm_time += vlm_elapsed
            self.total_upload_bytes += sent_bytes
            self.upload_frames += 1
            self.total_upload_crops += 1
            if boxes:
                best_box = select_best_candidate(boxes, self.last_success_bbox, (h, w))
                if best_box:
                    self.current_bbox = normalize_bbox(best_box, h, w)
                    self.tracker_initialized = False
                    self.last_success_bbox = self.current_bbox
                    logger.info("Cloud-Only: got box from cloud.")
                    return
            self.current_bbox = None
            self.tracker_initialized = False
            logger.warning("Cloud-Only: no boxes from cloud.")
            return

        if self.edge_only_mode:
            if dets:
                best = max(dets, key=lambda x: x[4])
                best_box = best[:4]
                self.current_bbox = normalize_bbox(best_box, h, w)
                init_xywh = [
                    self.current_bbox[0],
                    self.current_bbox[1],
                    self.current_bbox[2] - self.current_bbox[0],
                    self.current_bbox[3] - self.current_bbox[1]
                ]
                self.tracker.init(frame, init_xywh)
                self.tracker_initialized = True
                self.last_success_bbox = self.current_bbox
                self.kalman = KalmanBoxTracker(self.current_bbox)
                logger.info("Edge-only: initialized with YOLO best box.")
            else:
                logger.warning("Edge-only: no detection on first frame.")
                self.tracker_initialized = False
                self.current_bbox = None
            return

        if self.cloudtrack_mode:
            short_cat = self.fallback_category if self.fallback_category else "person"
            start_vlm = time.time()
            boxes, _, sent_bytes = self.vlm_client.detect_and_filter(
                frame, short_cat, self.text_prompt, self.overscan
            )
            vlm_elapsed = time.time() - start_vlm
            self.vlm_call_count += 1
            self.total_vlm_time += vlm_elapsed
            self.total_upload_bytes += sent_bytes
            self.upload_frames += 1
            self.total_upload_crops += 1
            if boxes:
                best_box = select_best_candidate(boxes, None, (h, w))
                if best_box:
                    self.current_bbox = normalize_bbox(best_box, h, w)
                    init_xywh = [
                        self.current_bbox[0],
                        self.current_bbox[1],
                        self.current_bbox[2] - self.current_bbox[0],
                        self.current_bbox[3] - self.current_bbox[1]
                    ]
                    self.tracker.init(frame, init_xywh)
                    self.tracker_initialized = True
                    self.last_success_bbox = self.current_bbox
                    self.kalman = KalmanBoxTracker(self.current_bbox)
                    logger.info("CloudTrack: initialized with cloud detection.")
                    return
            logger.warning("CloudTrack: no boxes, skip init.")
            self.tracker_initialized = False
            self.current_bbox = None
            return

        # ---------- ours 模式 ----------
        if not dets:
            logger.warning("No detections on first frame.")
            self.tracker_initialized = False
            self.current_bbox = None
            return

        if self.network_available:
            start_vlm = time.time()
            result, success = self._call_vlm(
                self.vlm_client.filter_boxes,
                frame, dets, self.text_prompt, self.overscan
            )
            if success and result is not None:
                matches, _, sent_bytes = result
                vlm_elapsed = time.time() - start_vlm
                self.vlm_call_count += 1
                self.total_vlm_time += vlm_elapsed
                self.total_upload_bytes += sent_bytes
                self.total_upload_crops += len(dets)
                self.upload_frames += 1

                valid_candidates = [dets[i] for i, m in enumerate(matches) if m and i < len(dets)]
                if valid_candidates:
                    best_box = select_best_candidate(valid_candidates, None, (h, w))
                    if best_box:
                        self._init_tracker_from_box(frame, best_box)
                        logger.info("Tracker initialized with VLM-verified box.")
                        return
                logger.warning("VLM verification succeeded but no match, skip initialization.")
                self.tracker_initialized = False
                self.current_bbox = None
                return
            logger.warning("VLM call failed, using YOLO box directly.")
            best_box = max(dets, key=lambda x: x[4])[:4]
            self._init_tracker_from_box(frame, best_box)
            logger.info("Tracker initialized with YOLO-only box (fallback).")
        else:
            logger.warning("Network unavailable, using YOLO box directly.")
            best_box = max(dets, key=lambda x: x[4])[:4]
            self._init_tracker_from_box(frame, best_box)
            logger.info("Tracker initialized with YOLO-only box (network degraded).")

    def _init_tracker_from_box(self, frame, box_xyxy):
        h, w = frame.shape[:2]
        self.current_bbox = normalize_bbox(box_xyxy, h, w)
        init_xywh = [
            self.current_bbox[0],
            self.current_bbox[1],
            self.current_bbox[2] - self.current_bbox[0],
            self.current_bbox[3] - self.current_bbox[1]
        ]
        self.tracker.init(frame, init_xywh)
        self.tracker_initialized = True
        self.last_success_bbox = self.current_bbox
        self.kalman = KalmanBoxTracker(self.current_bbox)
        self.anomaly_counter = 0

    # ---------- 重检测 ----------
    def _handle_re_detection(self, frame):
        self.re_detection_count += 1
        h, w = frame.shape[:2]

        if self.no_reid_mode:
            if self.kalman is not None:
                pred_box = self.kalman.get_predicted_bbox()
                self.current_bbox = normalize_bbox(pred_box, h, w)
                self.tracker_initialized = False
            else:
                self.current_bbox = None
                self.tracker_initialized = False
            return

        if self.cloud_only_mode:
            short_cat = self.fallback_category if self.fallback_category else "people"
            start_vlm = time.time()
            boxes, _, sent_bytes = self.vlm_client.detect_and_filter(
                frame, short_cat, self.text_prompt, self.overscan
            )
            vlm_elapsed = time.time() - start_vlm
            self.vlm_call_count += 1
            self.total_vlm_time += vlm_elapsed
            self.total_upload_bytes += sent_bytes
            self.upload_frames += 1
            self.total_upload_crops += 1

            if boxes:
                best_box = select_best_candidate(boxes, self.last_success_bbox, (h, w))
                if best_box:
                    self.current_bbox = normalize_bbox(best_box, h, w)
                    self.tracker_initialized = False
                    self.last_success_bbox = self.current_bbox
                    logger.info("Cloud-Only: got box from cloud.")
                    return
            self.current_bbox = None
            self.tracker_initialized = False
            logger.warning("Cloud-Only: no boxes from cloud.")
            return

        if not self.cloudtrack_mode:
            start_yolo = time.time()
            dets = self.yolo.detect(frame)
            self.yolo_time_total += time.time() - start_yolo
            self.yolo_call_count += 1
        else:
            dets = []

        if self.edge_only_mode:
            if dets:
                best_box = select_best_candidate(dets, self.last_success_bbox, (h, w))
                if best_box:
                    self._reinit_tracker(frame, best_box)
                    logger.info("Edge-only re-init with YOLO box.")
                    return
            self._fallback_to_kalman(h, w)
            return

        if self.cloudtrack_mode:
            short_cat = self.fallback_category if self.fallback_category else "person"
            start_vlm = time.time()
            boxes, _, sent_bytes = self.vlm_client.detect_and_filter(
                frame, short_cat, self.text_prompt, self.overscan
            )
            vlm_elapsed = time.time() - start_vlm
            self.vlm_call_count += 1
            self.total_vlm_time += vlm_elapsed
            self.total_upload_bytes += sent_bytes
            self.upload_frames += 1
            self.total_upload_crops += 1
            if boxes:
                best_box = select_best_candidate(boxes, self.last_success_bbox, (h, w))
                if best_box:
                    self._reinit_tracker(frame, best_box)
                    logger.info("CloudTrack re-init.")
                    return
            self.current_bbox = None
            self.tracker_initialized = False
            logger.warning("CloudTrack: no boxes from cloud, waiting for next frame.")
            return

        # ---------- ours 模式 ----------
        if not dets:
            if (not self.disable_fallback) and self.network_available:
                fallback_cat = self.fallback_category or "person"
                result, success = self._call_vlm(
                    self.vlm_client.detect_and_filter,
                    frame, fallback_cat, self.text_prompt, self.overscan
                )
                if success and result is not None:
                    boxes, _, sent_bytes = result
                    self.vlm_call_count += 1
                    self.total_vlm_time += time.time() - 0  # 计时已在外部处理
                    self.total_upload_bytes += sent_bytes
                    self.upload_frames += 1
                    self.total_upload_crops += 1
                    if boxes:
                        best_box = select_best_candidate(boxes, self.last_success_bbox, (h, w))
                        if best_box:
                            self._reinit_tracker(frame, best_box)
                            logger.info("Fallback CloudTrack re-init success.")
                            return
            self._fallback_to_kalman(h, w)
            return

        if self.network_available:
            start_vlm = time.time()
            result, success = self._call_vlm(
                self.vlm_client.filter_boxes,
                frame, dets, self.text_prompt, self.overscan
            )
            if success and result is not None:
                matches, _, sent_bytes = result
                vlm_elapsed = time.time() - start_vlm
                self.vlm_call_count += 1
                self.total_vlm_time += vlm_elapsed
                self.total_upload_bytes += sent_bytes
                self.total_upload_crops += len(dets)
                self.upload_frames += 1

                valid_candidates = [dets[i] for i, m in enumerate(matches) if m and i < len(dets)]
                if valid_candidates:
                    best_box = select_best_candidate(valid_candidates, self.last_success_bbox, (h, w))
                    if best_box:
                        self._reinit_tracker(frame, best_box)
                        logger.info("Re-init with VLM-verified box.")
                        return
                logger.warning("No VLM match, fallback to Kalman prediction.")
                self._fallback_to_kalman(h, w)
                return
            logger.warning("VLM call failed, fallback to YOLO-only.")
            best_box = select_best_candidate(dets, self.last_success_bbox, (h, w))
            if best_box:
                self._reinit_tracker(frame, best_box)
                logger.info("Re-init with YOLO-only (VLM failed).")
                return
            self._fallback_to_kalman(h, w)
        else:
            best_box = select_best_candidate(dets, self.last_success_bbox, (h, w))
            if best_box:
                self._reinit_tracker(frame, best_box)
                logger.info("Re-init with YOLO-only (network degraded).")
                return
            self._fallback_to_kalman(h, w)

    def _reinit_tracker(self, frame, box_xyxy):
        h, w = frame.shape[:2]
        self.current_bbox = normalize_bbox(box_xyxy, h, w)
        init_xywh = [self.current_bbox[0], self.current_bbox[1],
                     self.current_bbox[2]-self.current_bbox[0],
                     self.current_bbox[3]-self.current_bbox[1]]
        self.tracker.init(frame, init_xywh)
        self.tracker_initialized = True
        self.last_success_bbox = self.current_bbox
        self.kalman = KalmanBoxTracker(self.current_bbox)
        self.anomaly_counter = 0

    def _fallback_to_kalman(self, h, w):
        if self.kalman is not None:
            pred_box = self.kalman.get_predicted_bbox()
            self.current_bbox = normalize_bbox(pred_box, h, w)
            self.tracker_initialized = False
            logger.debug("Using Kalman prediction.")
        else:
            self.current_bbox = None
            self.tracker_initialized = False

    def _call_vlm(self, func, *args, force=False, **kwargs):
        if not self.network_available and not force:
            return None, False

        try:
            result = func(*args, **kwargs)
            self.network_fail_count = 0
            return result, True
        except Exception as e:
            logger.warning(f"VLM call failed: {e}")
            if self.network_available:
                self.network_fail_count += 1
                if self.network_fail_count >= self.max_failures:
                    self.network_available = False
                    self.degradation_count += 1
                    logger.error("Network deemed unavailable. Switching to edge-only mode.")
            return None, False

    def _probe_network(self, frame):
        if self.network_available:
            return
        h, w = frame.shape[:2]
        dets = self.yolo.detect(frame)
        if not dets:
            logger.debug("Probe: No detections, skip VLM filter.")
            return

        try:
            # 如果旧 socket 存在，先关闭
            if self.vlm_client.sock is not None:
                self.vlm_client.sock.close()
            # 重新连接
            self.vlm_client.connect()  # 这会创建一个新的 socket
            logger.info("Probe: TCP connection re-established.")
        except Exception as e:
            logger.debug(f"Probe: Reconnect failed: {e}")
            return

        
        result, success = self._call_vlm(
            self.vlm_client.filter_boxes,
            frame, dets, self.text_prompt, self.overscan,
            force=True
        )
        if success and result is not None:
            matches, _, _ = result
            self.network_available = True
            self.network_fail_count = 0
            self.recovery_count += 1
            logger.info("Network recovered! Switching back to normal mode.")

            valid_candidates = [dets[i] for i, m in enumerate(matches) if m and i < len(dets)]
            if valid_candidates:
                best_box = select_best_candidate(valid_candidates, self.last_success_bbox, (h, w))
                if best_box:
                    self._reinit_tracker(frame, best_box)
                    logger.info("Re-init with VLM-verified box (recovered).")
                    return
            logger.info("Probe success but no VLM match, network restored.")
        else:
            logger.debug("Probe failed, keeping degraded mode.")

# ================== 主程序 ==================
def main():
    parser = argparse.ArgumentParser(description="RKNN evaluation with multiple modes")
    parser.add_argument("--pc_ip", type=str, default="192.168.0.24")
    parser.add_argument("--data_root", type=str, default="/workspace/UAV123/sequences")
    parser.add_argument("--yolo_model", type=str, default="/workspace/best.rknn")
    parser.add_argument("--backbone_127", type=str, default="/workspace/nano_rknn_models/backbone_127.rknn")
    parser.add_argument("--backbone_255", type=str, default="/workspace/nano_rknn_models/backbone_255.rknn")
    parser.add_argument("--head_model", type=str, default="/workspace/nano_rknn_models/head.rknn")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--overscan", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="/workspace/eval_results.csv")
    parser.add_argument("--kalman_factor", type=float, default=4)
    parser.add_argument("--anomaly_frames", type=int, default=4)
    parser.add_argument("--seq_indices", type=str, default=None,
                        help="Comma-separated list of sequence indices")
    parser.add_argument("--mode", type=str, default="ours",
                        choices=["ours", "no_reid", "no_anomaly", "no_score",
                                 "cloudtrack", "edge_only", "cloud_only"],
                        help="Experiment mode")
    parser.add_argument("--fallback_category", type=str, default='people',
                    help="Short category for GroundingDINO fallback (e.g., 'person', 'car'). "
                         "If not provided, defaults to 'person'.")
    parser.add_argument("--disable_fallback", action="store_true",
                    help="Disable CloudTrack fallback when YOLO has no detections.")
    parser.add_argument("--save_video", action="store_true",
                    help="Save output video with GT and tracking boxes for each sequence.")
    parser.add_argument("--video_output_dir", type=str, default="/workspace",
                        help="Directory to save output videos.")
    args = parser.parse_args()

    seq_root = Path(args.data_root)
    seq_dirs = [p for p in seq_root.iterdir() if p.is_dir() and (p / "color").exists() and (p / "groundtruth.txt").exists()]
    seq_dirs.sort()
    total_seqs = len(seq_dirs)
    logger.info(f"Found {total_seqs} sequences.")

    if args.seq_indices is not None:
        selected_indices = [int(x.strip()) for x in args.seq_indices.split(",")]
        selected = [seq_dirs[i] for i in selected_indices if 0 <= i < total_seqs]
        if not selected:
            logger.error("No valid indices in --seq_indices")
            return
        logger.info(f"Processing specified indices: {selected_indices} (total {len(selected)})")
    else:
        start_idx = args.start
        end_idx = args.end if args.end is not None else total_seqs
        if start_idx < 0 or start_idx >= total_seqs:
            logger.error(f"start index {start_idx} out of range (0-{total_seqs-1})")
            return
        if end_idx > total_seqs or end_idx <= start_idx:
            logger.error(f"end index {end_idx} invalid")
            return
        selected = seq_dirs[start_idx:end_idx]
        logger.info(f"Processing sequences {start_idx} to {end_idx-1} (total {len(selected)})")

    for seq_path in selected:
        seq_name = seq_path.name
        logger.info(f"===== Evaluating sequence: {seq_name} (mode={args.mode}) =====")
        prompt = read_language_prompt(seq_path)
        logger.info(f"  Prompt: {prompt}")

        evaluator = SingleSequenceEvaluator(
            seq_path=str(seq_path),
            text_prompt=prompt,
            pc_ip=args.pc_ip,
            yolo_model_path=args.yolo_model,
            tracker_model_paths=(args.backbone_127, args.backbone_255, args.head_model),
            score_threshold=args.score_threshold,
            overscan=args.overscan,
            max_frames=args.max_frames,
            kalman_factor=args.kalman_factor,
            anomaly_frames=args.anomaly_frames,
            mode=args.mode,
            fallback_category=args.fallback_category,
            disable_fallback=args.disable_fallback,
            save_video=args.save_video,
            video_output_dir=args.video_output_dir
        )
        stats = evaluator.run()

        logger.info(f"  {seq_name}: MIoU={stats['miou']:.4f}, FPS={stats['fps']:.2f}, "
                    f"SR={stats['success_rate']:.4f}, ValidGT={stats['iou_count']}/{stats['total_frames']}, "
                    f"VLM_calls={stats['vlm_calls']}, VLM_time={stats['vlm_time']:.2f}s, "
                    f"ReDet={stats['re_detections']}, Lost={stats['lost_count']}")

        # 保存CSV
        try:
            import pandas as pd
            df_seq = pd.DataFrame([stats])
            if os.path.exists(args.output_csv):
                df_existing = pd.read_csv(args.output_csv)
                df_combined = pd.concat([df_existing, df_seq], ignore_index=True)
                df_combined.to_csv(args.output_csv, index=False)
            else:
                df_seq.to_csv(args.output_csv, index=False)
        except ImportError:
            logger.warning("pandas not installed, cannot save CSV")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    logger.info(f"Finished processing sequences.")

if __name__ == "__main__":
    main()