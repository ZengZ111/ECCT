from pathlib import Path
import cv2
import io
import groundingdino
import numpy as np
import torch
from groundingdino.datasets import transforms as T
from groundingdino.util.inference import load_model, predict
from PIL import Image

from .wrapper_base import WrapperBase


def letterbox_pil(image_pil, target_size=640):
    """
    Apply letterbox to PIL image: resize to target_size while preserving aspect ratio,
    pad with gray (114,114,114) to make square.
    Returns resized PIL image.
    """
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb)


def image_to_pil_and_image(image_pil: Image, use_letterbox=False, input_size=640):
    """
    Compatibility function, now optionally uses letterbox for aligned preprocessing.
    """
    if use_letterbox:
        image_pil = letterbox_pil(image_pil, target_size=input_size)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
    else:
        transform = T.Compose(
            [
                T.RandomResize([540], max_size=960),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)
    image_np = np.array(image_pil)
    return image_pil, image, image_np


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it."""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class GroundingDinoWrapper(WrapperBase):
    def __init__(
        self,
        cfg_in=None,
        use_sam_hq=True,
        box_threshold=0.3,
        text_threshold=0.25,
        args="",
        input_size=640,
        use_letterbox=True,
    ) -> None:
        cloudtrack_folder = Path(__file__).parent.parent.parent.resolve()
        dino_model_folder = cloudtrack_folder / "models/groundingdino"
        dino_pth_path = dino_model_folder / "groundingdino_swint_ogc.pth"
        dino_module_folder = Path(groundingdino.__file__).parent.resolve()
        dino_ogc_path = dino_module_folder / "config/GroundingDINO_SwinT_OGC.py"

        self.model = load_model(
            model_config_path=str(dino_ogc_path),
            model_checkpoint_path=str(dino_pth_path),
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.input_size = input_size
        self.use_letterbox = use_letterbox

    def _preprocess_image(self, image_pil):
        """Preprocess PIL image according to use_letterbox and input_size."""
        if self.use_letterbox:
            image_pil = letterbox_pil(image_pil, target_size=self.input_size)
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
        else:
            transform = T.Compose(
                [
                    T.RandomResize([540], max_size=960),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_transformed, _ = transform(image_pil, None)
        return image_transformed

    def run_inference_with_timing(
        self,
        image_pil: Image,
        prompt: str,
        nms_iou_thresh: float = None,   
        print_results=False,
        mark_results=False,
    ):
        
        import time
        import torch
        import torchvision
        from groundingdino.util.inference import predict
    
        timings = {}
    
        torch.cuda.synchronize()
        t_pre0 = time.perf_counter()
    
        image_transformed = self._preprocess_image(image_pil)
    
        torch.cuda.synchronize()
        t_pre1 = time.perf_counter()
        timings['preprocess'] = (t_pre1 - t_pre0) * 1000.0
    
        torch.cuda.synchronize()
        t_inf0 = time.perf_counter()
    
        boxes_raw, scores_raw, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=prompt + " .",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
    
        torch.cuda.synchronize()
        t_inf1 = time.perf_counter()
        timings['inference'] = (t_inf1 - t_inf0) * 1000.0
    
        torch.cuda.synchronize()
        t_post0 = time.perf_counter()
    
        if boxes_raw is not None:
            boxes = boxes_raw.cpu().numpy().copy()
            w, h = image_pil.size
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes = torch.tensor(boxes)          
        else:
            boxes = torch.zeros((0, 4))
    
        if nms_iou_thresh is not None and len(boxes) > 0:
            scores = torch.tensor(scores_raw, dtype=torch.float32)
            keep = torchvision.ops.nms(boxes, scores, nms_iou_thresh)
            boxes = boxes[keep]
            scores_raw = [scores_raw[i] for i in keep.tolist()]   
    
        masks = torch.zeros((len(boxes), 1, image_pil.size[1], image_pil.size[0])).bool()
    
        if mark_results and len(boxes) > 0:
            image_pil = self.visualize(image_pil, boxes, labels=phrases, masks=masks)
    
        boxes_list = boxes.cpu().numpy().tolist() if len(boxes) > 0 else []
        scores_list = [float(s) for s in scores_raw] if scores_raw is not None else []
    
        torch.cuda.synchronize()
        t_post1 = time.perf_counter()
        timings['postprocess'] = (t_post1 - t_post0) * 1000.0
    
        return image_pil, masks, boxes_list, scores_list, timings
        
    def run_inference(
        self,
        image_pil: Image,
        prompt: str,
        print_results=True,
        mark_results=False,
    ):
        """Runs Grounded DINO on the image and returns results."""
        image_transformed = self._preprocess_image(image_pil)

        boxes, scores, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=prompt + " .",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        masks = torch.zeros(
            (len(boxes), 1, image_pil.size[1], image_pil.size[0])
        ).bool()

        if mark_results:
            image_pil = self.visualize(
                image_pil, boxes, labels=phrases, masks=masks
            )

        if boxes is not None:
            boxes = boxes.cpu().numpy()
            w, h = image_pil.size
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # convert cxcywh to x1y1x2y2
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            if print_results:
                print(boxes, scores, phrases)

            boxes = torch.tensor(boxes)
            scores = scores.cpu().numpy()
            scores = [float(score) for score in scores]

        return image_pil, masks, boxes, scores