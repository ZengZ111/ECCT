#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
import pickle
import argparse
from rknn_tracker import RKNNTracker  # 确保该文件存在
from rknnlite.api import RKNNLite

def save_tracker_state(tracker, state_file):
    """保存 tracker 的核心状态（不包含模型对象）"""
    state = {
        'center_pos': tracker.center_pos,
        'size': tracker.size,
        'zf': tracker.zf,
        'channel_average': tracker.channel_average,
        'params': tracker.params,
        'window': tracker.window,
        'points': tracker.points,
        'score_size': tracker.score_size,
        'cls_out_channels': tracker.cls_out_channels
    }
    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

def load_tracker_state(state_file, tracker):
    """从状态文件恢复 tracker 的核心状态"""
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    tracker.center_pos = state['center_pos']
    tracker.size = state['size']
    tracker.zf = state['zf']
    tracker.channel_average = state['channel_average']
    tracker.params = state['params']
    tracker.window = state['window']
    tracker.points = state['points']
    tracker.score_size = state['score_size']
    tracker.cls_out_channels = state['cls_out_channels']
    return tracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['init', 'track'], required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--bbox', nargs=4, type=float, help='[x,y,w,h] for init')
    parser.add_argument('--state', required=True)
    args = parser.parse_args()

    frame = cv2.imread(args.img)
    if frame is None:
        print(json.dumps({"error": "cannot read image"}))
        return 1

    if args.mode == 'init':
        # 创建新的 tracker（需要模型路径）
        tracker = RKNNTracker(
            '/workspace/nano_rknn_models/backbone_127.rknn',
            '/workspace/nano_rknn_models/backbone_255.rknn',
            '/workspace/nano_rknn_models/head.rknn',
            core_mask=RKNNLite.NPU_CORE_0_1_2
        )
        bbox = [float(x) for x in args.bbox]
        tracker.init(frame, bbox)
        save_tracker_state(tracker, args.state)
        print(json.dumps({"status": "ok"}))
        return 0

    elif args.mode == 'track':
        # 重建 tracker（需要模型路径）
        tracker = RKNNTracker(
            '/workspace/nano_rknn_models/backbone_127.rknn',
            '/workspace/nano_rknn_models/backbone_255.rknn',
            '/workspace/nano_rknn_models/head.rknn',
            core_mask=RKNNLite.NPU_CORE_0_1_2
        )
        # 加载状态
        tracker = load_tracker_state(args.state, tracker)
        result = tracker.track(frame)
        # 更新状态文件
        save_tracker_state(tracker, args.state)
        print(json.dumps({"bbox": result['bbox'], "score": result['score']}))
        return 0

if __name__ == "__main__":
    sys.exit(main())