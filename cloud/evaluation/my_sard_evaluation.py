from utils.single_shot_pipeline import ExperimentPipeline
from pathlib import Path
import sys

result_path = Path(sys.argv[1])

pipeline = ExperimentPipeline(result_path)

ap = pipeline.get_average_precision(
    iou_threshold=0.5,
    skip_vlm_check=False
)

precision, recall = pipeline.get_precision_recall(
    iou_threshold=0.5,
    skip_vlm_check=False,
    conf_threshold=-1
)

tp, fp, tn, fn = pipeline.get_confusion_matrix(
    iou_threshold=0.5,
    skip_vlm_check=False
)


print(f"AP50 = {ap*100:.2f}%")
print(f"Precision = {precision*100:.2f}%")
print(f"Recall    = {recall*100:.2f}%")



print(f"TP = {tp}")
print(f"FP = {fp}")
print(f"FN = {fn}")

# =====================================
# Time Statistics
# =====================================

total_crop_time = 0
wifi_upload = 0
g4_upload = 0
g5_upload = 0
total_vlm_time = 0
total_detector_time = 0

total_objects = 0

for frame in pipeline.frames:

    frame_data = pipeline.res[frame]

    total_crop_time += frame_data.get(
        "crop_time_s", 0
    )

    wifi_upload += frame_data.get(
        "wifi_upload_time_s", 0
    )

    g4_upload += frame_data.get(
        "4g_upload_time_s", 0
    )

    g5_upload += frame_data.get(
        "5g_upload_time_s", 0
    )

    total_vlm_time += frame_data.get(
        "vlm_time_s", 0
    )

    total_detector_time += frame_data.get(
        "detector_time_s", 0
    )

    total_objects += frame_data.get(
        "num_objects", 0
    )

num_frames = len(pipeline.frames)

print("\n========== Timing ==========")

print(
    f"Average Crop Time / Frame = "
    f"{total_crop_time / num_frames:.6f} s"
)

print(
    f"Average WiFi Upload Time / Frame = "
    f"{wifi_upload / num_frames:.6f} s"
)

print(
    f"Average 4G Upload Time / Frame = "
    f"{g4_upload / num_frames:.6f} s"
)

print(
    f"Average 5G Upload Time / Frame = "
    f"{g5_upload / num_frames:.6f} s"
)

print(
    f"Average VLM Time / Frame = "
    f"{total_vlm_time / num_frames:.6f} s"
)

print(
    f"Average Detector Time / Frame = "
    f"{total_detector_time / num_frames:.6f} s"
)

if total_objects > 0:

    print(
        f"Average Crop Time / Object = "
        f"{total_crop_time / total_objects:.6f} s"
    )

    print(
        f"Average WiFi Upload Time / Object = "
        f"{wifi_upload / total_objects:.6f} s"
    )
    
    print(
        f"Average 4G Upload Time / Object = "
        f"{g4_upload / total_objects:.6f} s"
    )
    
    print(
        f"Average 5G Upload Time / Object = "
        f"{g5_upload / total_objects:.6f} s"
    )

    print(
        f"Average VLM Time / Object = "
        f"{total_vlm_time / total_objects:.6f} s"
    )