#!/usr/bin/env python3
import sys
import json
import cv2
from rknn_yolo import RKNNYOLO

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing image path"}))
        return 1
    img_path = sys.argv[1]
    frame = cv2.imread(img_path)
    if frame is None:
        print(json.dumps({"error": "cannot read image"}))
        return 1

    yolo = RKNNYOLO('/workspace/best.rknn', conf_thres=0.3, iou_thres=0.45, imgsz=640, num_classes=5)
    boxes = yolo.detect(frame)
    print(json.dumps({"boxes": boxes}))
    yolo.release()
    return 0

if __name__ == "__main__":
    sys.exit(main())