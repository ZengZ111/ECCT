from pathlib import Path
import re
import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase
from cloud_track.rpc_communication.utils import deserialize
from .gpt_four_wrapper import GPTFourWrapper
from .grounding_dino_huggingface_wrapper import GroundingDinoHuggingfaceWrapper
from .grounding_dino_wrapper import GroundingDinoWrapper
from .llava_wrapper import LlavaWrapper
from .paligemma_wrapper import PaligemmaWrapper
from .yolo_wrapper import YoloWrapper

import time
import io



class DetectorVlmPipeline(WrapperBase):
    def __init__(
        self,
        vlm: WrapperBase,
        detector: WrapperBase,
        enable_overscan=True,
        overscan_value=50,
    ):
        """
        Instantiates the VLM pipeline. This pipeline combines the GPTFourWrapper
        and the GroundedSamWrapper to find arbitrary objects in images.

        Args:
            role_desctiption (str): Description of the system. e.g. "You are a
                drone on a search and rescue mission." This becomes a part of
                the chatGPT system prompt.
            enable_caching (bool, optional): _description_. Defaults to True.
            simulate_time_delay (bool, optional): _description_. Defaults to False.
            use_sam_hq (bool, optional): _description_. Defaults to True.
            enable_overscan (bool, optional): _description_. Defaults to True.
            gpt_model (str, optional): _description_. Defaults to "gpt-4o".
        """
        self.vlm = vlm
        self.detector = detector

        self.enable_overscan = enable_overscan
        self.overscan_value = 50  # pixels in each direction

        self.debug_enable_gpt = True


    def parse_vlm_response(self, reply):
    
        reply_lower = reply.lower()
    

        if "answer: yes" in reply_lower:
            match = True
        elif "answer: no" in reply_lower:
            match = False
        elif reply_lower.strip().startswith("yes"):
            match = True
        elif reply_lower.strip().startswith("no"):
            match = False
        elif " yes " in reply_lower:
            match = True
        elif " no " in reply_lower:
            match = False
        else:
            logger.warning(
                f"Could not parse yes/no from response: {reply}"
            )
            match = False
    
        # justification
        if "justification:" in reply_lower:
            justification = reply.split("Justification:")[-1].strip()
        else:
            justification = reply.strip()
    
        return match, justification
        
    

    def parse_vlm_response_list(self, gpt_responses: list[str]):
        """Parases a list of gpt responses. For each response, it parses the

        Args:
            gpt_responses list[str]: A list of GPT responses as text.

        Returns:
            list[bool]: A list of booleans indicating if the response is a match.
            list[str]: A list of justifications for the responses.
        """

        matches = []
        justifications = []

        for response in gpt_responses:
            match, justification = self.parse_vlm_response(response)
            matches.append(match)
            justifications.append(justification)

        return matches, justifications

    def run_inference_inner(
        self, cathegory: str, verbal_description: str, image: Image
    ):
        if self.vlm is None:
            # we give the better prompt to the detector if no VLM is available
            cathegory = verbal_description
    
        detector_start = time.perf_counter()

        image_pil, masks, boxes_filt, scores = self.detector.run_inference(
            image,
            prompt=cathegory,
            mark_results=False
        )
        
        detector_time = time.perf_counter() - detector_start
    
        prompt = verbal_description
        vlm_responses = []
        
        total_crop_time = 0
        total_wifi_upload_time = 0
        total_4g_upload_time = 0
        total_5g_upload_time = 0
        total_vlm_time = 0
    
        for row in boxes_filt:
            # 1. 转换为整数
            row = [int(x) for x in row]
            x1, y1, x2, y2 = row
    
            # 2. 确保 x1 <= x2, y1 <= y2 (修正坐标颠倒)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
    
            # 3. 扩展边界 (overscan)
            if self.enable_overscan:
                x1 = max(0, x1 - self.overscan_value)
                y1 = max(0, y1 - self.overscan_value)
                x2 = min(image.width, x2 + self.overscan_value)
                y2 = min(image.height, y2 + self.overscan_value)
    
            # 4. 再次裁剪到图像边界内
            x1 = max(0, min(x1, image.width))
            x2 = max(0, min(x2, image.width))
            y1 = max(0, min(y1, image.height))
            y2 = max(0, min(y2, image.height))
    
            # 5. 检查矩形是否有效
            if x1 < x2 and y1 < y2:
                crop_start = time.perf_counter()
                
                cropped_image = image.crop((x1,y1,x2,y2))
                
                crop_time = time.perf_counter() - crop_start
                
                total_crop_time += crop_time
                
                buffer = io.BytesIO()
                
                cropped_image.save(
                    buffer,
                    format="JPEG",
                    quality=90
                )
                
                image_size_bytes = len(buffer.getvalue())
                
                
                # =====================================
                # WiFi
                # =====================================
                
                wifi_bandwidth = 20 * 1024 * 1024 / 8
                
                wifi_upload_time = (
                    image_size_bytes / wifi_bandwidth
                ) + 0.020
                
                
                # =====================================
                # 4G
                # =====================================
                
                g4_bandwidth = 10 * 1024 * 1024 / 8
                
                g4_upload_time = (
                    image_size_bytes / g4_bandwidth
                ) + 0.050
                
                
                # =====================================
                # 5G
                # =====================================
                
                g5_bandwidth = 100 * 1024 * 1024 / 8
                
                g5_upload_time = (
                    image_size_bytes / g5_bandwidth
                ) + 0.010
                
                
                total_wifi_upload_time += wifi_upload_time
                total_4g_upload_time += g4_upload_time
                total_5g_upload_time += g5_upload_time
                
                logger.info(f"Found {cathegory} at {row}. Running VLM.")
                if not self.debug_enable_gpt:
                    raise NotImplementedError(
                        "GPTFourWrapper is disabled to save tokens during debug."
                    )
                logger.info(f"Prompt: {prompt}")
                if self.vlm is not None:
                    vlm_start = time.perf_counter()
                    
                    vlm_response = self.vlm.run_inference(
                        prompt,
                        cropped_image
                    )
                    
                    vlm_time = time.perf_counter() - vlm_start
                    
                    total_vlm_time += vlm_time
                else:
                    vlm_response = """
                    Answer: Yes
                    Justification: No vlm - detector only.
                    """
            else:
                # 无效框：给出默认响应，不裁剪图像
                logger.warning(f"Invalid bounding box after processing: {(x1, y1, x2, y2)}. Skipping crop and using default response.")
                vlm_response = """
                Answer: No
                Justification: Invalid bounding box from detector.
                """
    
            try:
                logger.info(f"VLM response: {vlm_response}")
            except KeyError:
                logger.warning("VLM Response has unexpected format.")
    
            vlm_responses.append(vlm_response)
    
        return (
            vlm_responses,
            image_pil,
            masks,
            boxes_filt,
            scores,
        
            detector_time,
            total_crop_time,
        
            total_wifi_upload_time,
            total_4g_upload_time,
            total_5g_upload_time,
        
            total_vlm_time
        )

    def run_inference(
        self,
        image: Image,
        category: str,
        description: str = None,
        mark_results=False,
        filter_results=False,
    ):
        filter_results = False 
        """Runs the inference pipeline. Parses the GPT response. If one of the
        objects in the image is a match, returns the bounding box of this
        object.

        Args:
            category (str): The category of the object to find
                (e.g. person, car, etc.)
            description (str): The instruction for the GPT model. (e.g. "Is this a person?")
            image (Image): The image to analyze

        Returns:
            _type_: _description_
        """
        # If no prompt is provided: construct a default prompt
        if not description:
            if " a " in category:
                description = f"Is this {category}?"
            else:
                description = f"Is this a {category}?"

        # If // in the cathegory: split here and use second half for description
        if "//" in category:
            l = category.split("//")
            category = l[0]
            description = l[1]

        # Run the actual inference:
        (
            gpt_responses,
            image_pil,
            masks,
            boxes_filt,
            scores,
        
            detector_time,
            total_crop_time,
        
            wifi_upload_time,
            g4_upload_time,
            g5_upload_time,
        
            total_vlm_time
        
        ) = self.run_inference_inner(
            category,
            description,
            image
        )

        # Postprocess the results:
        matches, justifications = self.parse_vlm_response_list(gpt_responses)

        # assemble the labels for the boxes
        labels = []
        for i, match in enumerate(matches):
            if match:
                labels.append(f"{category} (match) | {justifications[i]}")
            else:
                labels.append(f"{category} | {justifications[i]}")

        if mark_results:
            # visualize results
            image_pil = self.visualize(image_pil, boxes_filt, labels=labels)

        if filter_results:
            # filter the boxes, scores, masks, justification based on the matches
            to_filter = [masks, boxes_filt, scores, justifications]
            for idx, list_ in enumerate(to_filter):
                list_ = [list_[i] for i, match in enumerate(matches) if match]
                to_filter[idx] = list_

            masks, boxes_filt, scores, justifications = to_filter

            if len(masks) == 0:
                logger.info("No matches found.")
                return image_pil, None, None, None, None

            boxes_filt = torch.stack(boxes_filt)
            masks = torch.stack(masks)

            masks = None  # not a good solution but fixing the error would take too long
            return image_pil, masks, boxes_filt, scores, justifications

        else:
            return (
                image_pil,
                masks,
                boxes_filt,
                scores,
                matches,
                labels,
                justifications,
            
                detector_time,
                total_crop_time,
            
                wifi_upload_time,
                g4_upload_time,
                g5_upload_time,
            
                total_vlm_time
            )
    def run_vlm_only(self, image: Image, description: str):
        logger.info(f"run_vlm_only received image type: {type(image)}")
        if isinstance(image, dict) and "pickle" in image:
            image = deserialize(image)
            logger.info(f"Deserialized image to type: {type(image)}")
        if not isinstance(image, Image.Image):
            logger.error(f"Image is not PIL.Image, but {type(image)}")
            return False, f"Invalid image type: {type(image)}"
        if self.vlm is None:
            logger.warning("No VLM available, returning default.")
            return False, "No VLM available."
        response = self.vlm.run_inference(description, image)  # 或 (image, description)
        match, justification = self.parse_vlm_response(response)
        return match, justification
    def run_vlm_only_batch(self, images: list, description: str):
        results = []
        for img in images:
            match, justification = self.run_vlm_only(img, description)
            results.append((match, justification))
        return results


def system_prompt_from_description(description: str):
    """
    Helper function to generate a system prompt from a description. Here, we
    conduct some prompt engineering to make the GPT model more effective.
    """

    prompt = f"""{description} Return your answer in the following format:
    Answer: Yes/No
    Justification: [Your justification here]
    """

    return prompt


if __name__ == "__main__":

    output_folder = Path("~/Downloads/flextrack_gpt_output").expanduser()
    output_folder.mkdir(exist_ok=True)

    pipeline = DetectorVlmPipeline()
    opject_cathegory = "person"
    description = "You are looking for a person with a gray shirt, who is missing after beeing injured."
    search_and_rescue_desctiption = "You are on a search and rescue mission. {description} Is this person in the image?"

    image_folder = Path(__file__).parent / "images"

    answers = []
    for image_path in image_folder.glob("*.jpg"):
        logger.info(f"Processing image {image_path.name}")
        image = Image.open(image_path).convert("RGB")  # droop alpha channel

        res = pipeline.run_inference(
            category=opject_cathegory,
            description=search_and_rescue_desctiption,
            image=image,
        )

        image_annotated, *_ = res

        # save the images to the output folder
        image_annotated.save(output_folder / image_path.name)

        # show image with cv2
        cv_image = np.array(image_annotated)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv_image)
        cv2.waitKey(10)

        # input("Press Enter to continue...")


def get_detector(detector_model):
    box_threshold = 0.2
    detector_model = detector_model.lower()
    if "sam" in detector_model:
        if "hq" in detector_model:
            use_sam_hq = True
            detector = GroundingDinoWrapper(
                box_threshold=box_threshold, use_sam_hq=use_sam_hq
            )
        elif "lq" in detector_model:
            box_threshold = 0.1
            text_threshold = 0.05
            detector = GroundingDinoHuggingfaceWrapper(
                box_threshold=box_threshold, text_threshold=text_threshold
            )
        else:
            raise ValueError(f"Unknown SAM model {detector_model}.")
            
    elif "yolo" in detector_model:
        detector = YoloWrapper(
            "/home/zz/workspace/CloudTrack/cloud_track/yolo/runs/detect/runs/yolov8n_BUU_SARD/weights/best.pt"
        )

    elif "glee" in detector_model:
        split = detector_model.split("_")
        if len(split) == 1:
            raise ValueError(
                f"Unknown GLEE model {detector_model}: Please specify model name like GLEE_[lite/plus/pro]."
            )
        else:
            from .glee_wrapper import GLEEWrapper

            detector = GLEEWrapper(
                model_name=split[1], box_threshold=box_threshold
            )
    else:
        return None
    return detector


def get_vlm(vl_model, system_description, simulate_time_delay):
    print(f"get_vlm called with vl_model={vl_model}")
    if "gpt" in vl_model:
        system_prompt = system_prompt_from_description(system_description)
        vlm = GPTFourWrapper(
            enable_caching=False,
            simulate_time_delay=simulate_time_delay,
            model=vl_model,
            system_prompt=system_prompt,
            cache_file_name="sard_single_shot_cache.json",
        )
    elif "paligemma" in vl_model:
        vlm = PaligemmaWrapper(system_prompt=system_description)
    elif "llava" in vl_model:
        system_prompt = system_description
        vlm = LlavaWrapper(system_prompt=system_prompt, model_name="/home/zz/models/llava-1.5-13b-hf")
    else:
        vlm = None
    return vlm


def get_vlm_pipeline(
    vl_model_name: str,
    system_description: str,
    simulate_time_delay: bool,
    detector_name: str,
    openai_api_key: str = None,
):
    """Creates a VLM pipeline with a detector and a VLM model.

    Args:
        vl_model_name (str): Name of the VLM Model. We support:
            - gpt-4o-mini, gpt-4o, gpt-4-turbo
            - llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf
            - paligemma
        system_description(str)): The system prompt for the VLM model.
        simulate_time_delay (bool): GPT answers can be cached. Retrieving from
            the cache is faster then the API. When simulate_time_delay is True,
            a time.sleep() is applied to simulate the API delay.
        detector_name (str): Name of the detector model. We support:
            - sam_hq, sam_lq
        openai_api_key (str, optional): The openai api key. Defaults to None.

    Raises:
        ValueError: When an unknown model is requested.

    Returns:
        DetectorVLMPipeline: A configured VLM pipeline.
    """
    # models
    vlm = get_vlm(vl_model_name, system_description, simulate_time_delay)
    detector = get_detector(detector_name)

    if detector is None:
        raise ValueError(
            f"Unknown model {detector_name} - cannot run without detector."
        )

    if vlm is None:
        logger.info(
            f"No VLM or unknwon name {vl_model_name} specified. Running without VLM."
        )

    model = DetectorVlmPipeline(vlm, detector, overscan_value=50)

    return model
