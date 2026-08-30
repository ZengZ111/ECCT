from ultralytics import YOLO
from PIL import Image
import torch

from .wrapper_base import WrapperBase


class YoloWrapper(WrapperBase):

    def __init__(
        self,
        model_path,
        conf_threshold=0.3
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def run_inference(
        self,
        image_pil: Image,
        prompt=None,
        print_results=True,
        mark_results=False,
    ):

        result = self.model(
            image_pil,
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )[0]

        boxes = []
        scores = []

        for box in result.boxes:

            cls = int(box.cls)

            # person Àà
            if cls not in [0]:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            boxes.append(xyxy)

            scores.append(
                float(box.conf.cpu())
            )

        if len(boxes) > 0:
            boxes = torch.tensor(boxes)
        else:
            boxes = torch.empty((0,4))

        masks = torch.zeros(
            (
                len(boxes),
                1,
                image_pil.size[1],
                image_pil.size[0]
            )
        ).bool()

        return (
            image_pil,
            masks,
            boxes,
            scores
        )