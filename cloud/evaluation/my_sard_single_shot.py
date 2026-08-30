from omegaconf import DictConfig
import hydra
import glob
from pathlib import Path
from PIL import Image
from cloud_track.foundation_model_wrappers import detector_vlm_pipeline, gpt_four_wrapper, grounding_dino_huggingface_wrapper, paligemma_wrapper, llava_wrapper, glee_wrapper, grounding_dino_wrapper
import xmltodict
import json
from loguru import logger
import time
from tqdm import tqdm
try:
    from evaluation.utils.fix_logging import fix_logging, handle_exception
except ImportError:
    logger.warning("fix_logging not available, using dummy functions")
    def fix_logging():
        pass
    def handle_exception(exc_type, exc_value, exc_traceback):
        pass
import sys

import numpy as np

def convert_to_serializable(obj):
   # """递归地将 NumPy 类型转换为 Python 原生类型，以便 JSON 序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def get_idxs(path):
    # find all .jpg files in the directory
    all_files = glob.glob(f"{str(path)}/*.jpg", recursive=True)
    all_files.sort(key=lambda f: int(Path(f).stem))

    stems = [Path(f).stem for f in all_files]
    return stems


def load_image_and_annotation(file_path, xml_path):
    # load s.png image as PIL from the current directory
    image = Image.open(file_path)

    # load xml
    try:
        with open(xml_path, "r") as f:
            xml = f.read()
            parsed = xmltodict.parse(xml)
            annotation = parsed["annotation"]["object"]
            if not isinstance(annotation, list):
                annotation = [annotation]
    except (KeyError, xmltodict.expat.ExpatError) as e:
        logger.warning(f"Failed to parse {xml_path}: {e}")
        annotation = []   # 空列表表示无有效标注

    return image, annotation


@hydra.main(config_path="../conf", config_name="sard_single_shot_evaluation")
@logger.catch  # So kriegen wir die Exceptions in die Logdatei!!!
def main(cfg: DictConfig):
    """
    Running the evaluation on the SARD dataset in a single shot manner. We parse 
    every image through the DetectorVlmPipeline. We then ask, if the object is 
    hurt or injured. we treat the classes "seated and "laying_down" as "injured".
    All others are treated as "not injured".

    Args:
        cfg (DictConfig): _description_
    """
    fix_logging()

    # paths
    hydra_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    sard_dir = Path("/home/zz/workspace/CloudTrack/cloud_track/datasets/YQKJ_SARD_my_test/")

    # config
    save_every = 10
    benchmark = True
    experiment_name = cfg.experiment  # could be sar_shirt, sar_injury or sar_person
    vl_model_name = cfg.vlm.name
    detector_model = cfg.detector

    # log vl_model_name and exp
    logger.info(
        f"Running {experiment_name} with {vl_model_name} and {detector_model}")

    # get prompts from config
    category = "person"  # war mal person. human ...
    exp_name_key = f"{experiment_name}_prompt"
    system_description = cfg.vlm.prompt_set["system_prompt"]
    prompt = cfg.vlm.prompt_set[exp_name_key]

    # auto config
    simulate_time_delay = False
    render_images = False
    frame_limit = 5
    if benchmark:
        simulate_time_delay = True
        render_images = False
        frame_limit = -1

    # create model
    model = detector_vlm_pipeline.get_vlm_pipeline(vl_model_name, system_description,
                      simulate_time_delay, detector_model)

    logger.info(f"Logging to {hydra_dir}")
    '''
    # make image dir, delete if exists
    image_dir = hydra_dir / "images"
    if image_dir.exists():
        for file in image_dir.glob("*"):
            file.unlink()
    image_dir.mkdir()
    '''
    result_dict = {}
    result_dict["metadata"] = {
        "category": category,
        "system_description": system_description,
        "user": prompt,
        "vl_model": vl_model_name,
        "detector": detector_model
    }

    names = get_idxs(sard_dir)

    count = 0
    RK3588_SCALE = 15
    
    for name in tqdm(names):
        file_path = sard_dir / f"{name}.jpg"
        xml_path = sard_dir / f"{name}.xml"

        if not xml_path.exists():
            logger.warning(f"Skipping {name} due to missing gt file.")
            continue

        image, annotations = load_image_and_annotation(file_path, xml_path)

        start = time.time()
        # run inference

        (
            image_pil,
            masks,
            boxes_filt,
            scores,
            matches,
            labels,
            justifications,
        
            detector_time,
            crop_time,
        
            wifi_upload_time,
            g4_upload_time,
            g5_upload_time,
        
            vlm_time
        
        ) = model.run_inference(
            image=image,
            category=category,
            description=prompt,
            filter_results=False,
            mark_results=render_images
        )
        end = time.time()

        results = []
        for box, match, score, label in zip(boxes_filt, matches, scores, labels):
            results.append({
                "box": box.tolist(),
                "score": score,
                "match": match,  # if true: cathegory: yes - gpt: yes. ELSE: cathegory: yes - gpt: no
                "label": label
            })

        result_dict[name] = {
        
            "image": f"{name}.jpg",
        
            "result": results,
        
            "annotation": annotations,
            
            "time_s": start - end,
        
            "detector_time_s": detector_time,
        
            "crop_time_s": crop_time,
        
            "wifi_upload_time_s": wifi_upload_time,
        
            "4g_upload_time_s": g4_upload_time,
        
            "5g_upload_time_s": g5_upload_time,
        
            "vlm_time_s": vlm_time,
        
            "num_objects": len(boxes_filt)
        }
        '''
        # save image
        if render_images:
            image_pil.save(image_dir / f"{name}.jpg")
        '''
        count += 1
        if count % save_every == 0:
            serializable_dict = convert_to_serializable(result_dict)
            with open(hydra_dir / "sard_single_shot.json", "w") as f:
                json.dump(serializable_dict, f, indent=4)

        
        if count == frame_limit:
            logger.warning(f"Terminated due to frame limit.")
            break    

    serializable_dict = convert_to_serializable(result_dict)
    with open(hydra_dir / "sard_single_shot.json", "w") as f:
        json.dump(serializable_dict, f, indent=4)

    logger.info("Done.")

if __name__ == "__main__":
    main()
