
import hashlib
import os
import json
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI
from typing import Dict, List, Tuple

# --- 配置部分 ---
IMAGES_DIR = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\images\\test"
LABELS_DIR = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\labels\\test"
OUTPUT_FILE = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\annotations.json"
CACHE_DIR = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\vlm_cache"  
# 阿里云百炼API配置
YOUR_API_KEY = ""  # 替换为你的真实API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼OpenAI兼容endpoint
MODEL_NAME = "qwen3.7-plus"     # 使用的模型名称

# 初始化OpenAI客户端（复用连接，降低成本）
client = OpenAI(api_key=YOUR_API_KEY, base_url=BASE_URL)

# 确保缓存目录存在
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ---------- 1. 优化后的Prompt（提升准确率）----------
ANNOTATION_PROMPT = """
You are an expert image annotator. Analyze the person in the given cropped image.

Respond with a **valid JSON object only**, using the exact keys and value options.

**Shirt color** must be one of: blue, green, gray, others.
**Pose** must be one of: Stands, Seated, laying_down, Others.
**Injured** must be 0 or 1 (0 = not injured, 1 = injured).

Example output:
{"shirt_color": "blue", "pose": "Stands", "injured": 0}
"""

# ---------- 辅助函数 ----------
def get_cache_key(img_path: str, bbox: Tuple[int, int, int, int]) -> str:
    """根据图片路径和边界框坐标生成唯一的缓存键"""
    data = f"{img_path}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    return hashlib.md5(data.encode()).hexdigest()

def load_cached_result(cache_key: str) -> Optional[Dict]:
    """从本地缓存读取标注结果"""
    cache_file = Path(CACHE_DIR) / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_cached_result(cache_key: str, result: Dict):
    """保存标注结果到本地缓存"""
    cache_file = Path(CACHE_DIR) / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(result, f)

def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    """YOLO归一化坐标 -> 像素坐标 (xmin, ymin, xmax, ymax)"""
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    w_abs = width * img_w
    h_abs = height * img_h
    xmin = int(x_center_abs - w_abs / 2)
    ymin = int(y_center_abs - h_abs / 2)
    xmax = int(x_center_abs + w_abs / 2)
    ymax = int(y_center_abs + h_abs / 2)
    # 边界裁剪
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return xmin, ymin, xmax, ymax

def call_qwen_for_annotation(cropped_img: Image.Image) -> Dict:
    """调用Qwen3.7-Plus对单人图进行标注（使用优化后的prompt）"""
    # 将PIL图片转为base64
    import io
    img_bytes = io.BytesIO()
    cropped_img.save(img_bytes, format="JPEG")
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert image annotator. Output only valid JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": ANNOTATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            temperature=0.2,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        # 提取JSON（移除可能的markdown标记）
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        result = json.loads(content)
        # 确保字段完整
        return {
            "shirt_color": result.get("shirt_color", "unknown"),
            "pose": result.get("pose", "Others"),
            "injured": int(result.get("injured", 0))
        }
    except Exception as e:
        print(f"  [API错误] {e}")
        return {"shirt_color": "error", "pose": "Others", "injured": 0}

def load_existing_results() -> Dict:
    """加载已有的输出JSON，用于断点续传"""
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_results(results: Dict):
    """保存结果到JSON文件（原子写入，防止损坏）"""
    tmp_file = OUTPUT_FILE + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_file, OUTPUT_FILE)  # 原子替换

# ---------- 主处理流程 ----------
def main():
    # 加载已有结果（断点续传）
    all_results = load_existing_results()
    processed_images = set(all_results.keys())
    
    labels_path = Path(LABELS_DIR)
    images_path = Path(IMAGES_DIR)
    
    # 收集所有待处理的标签文件
    label_files = sorted(labels_path.glob("*.txt"))
    total = len(label_files)
    
    for idx, label_file in enumerate(label_files, 1):
        img_name = label_file.stem + ".jpg"
        img_path = images_path / img_name
        
        # 断点续传：如果该图片已经处理过且结果非空，则跳过
        if img_name in processed_images:
            print(f"[{idx}/{total}] 跳过已处理: {img_name}")
            continue
        
        if not img_path.exists():
            print(f"[{idx}/{total}] 警告: 图片不存在 {img_path}，跳过")
            continue
        
        # 读取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"[{idx}/{total}] 无法打开图片 {img_name}: {e}，跳过")
            continue
        
        # 读取标签文件（YOLO格式：每行 x_center y_center width height）
        bboxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        bboxes.append(tuple(map(float, parts)))
                    except ValueError:
                        continue
        
        if not bboxes:
            print(f"[{idx}/{total}] {img_name} 无有效标注框，跳过")
            continue
        
        print(f"[{idx}/{total}] 处理: {img_name}，共 {len(bboxes)} 个目标")
        
        img_annotations = {}
        for obj_idx, (x_c, y_c, w, h) in enumerate(bboxes):
            # 转换坐标
            xmin, ymin, xmax, ymax = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
            if xmin >= xmax or ymin >= ymax:
                print(f"  警告: 无效框索引 {obj_idx}，跳过")
                continue
            
            # 检查缓存（基于图片路径和边界框）
            cache_key = get_cache_key(str(img_path), (xmin, ymin, xmax, ymax))
            cached = load_cached_result(cache_key)
            
            if cached:
                attrs = cached
                print(f"  目标 {obj_idx}: 使用缓存")
            else:
                # 裁剪子图
                with Image.open(img_path) as img:
                    cropped = img.crop((xmin, ymin, xmax, ymax))
                # 调用API
                attrs = call_qwen_for_annotation(cropped)
                # 存入缓存
                save_cached_result(cache_key, attrs)
                print(f"  目标 {obj_idx}: API调用完成 -> 颜色={attrs['shirt_color']}, 姿态={attrs['pose']}")
            
            img_annotations[str(obj_idx)] = {
                "bndbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                "injured": attrs["injured"],
                "shirt_color": attrs["shirt_color"],
                "pose": attrs["pose"]
            }
        
        # 将当前图片的结果合并到总结果
        all_results[img_name] = img_annotations
        # 每处理完一张图片就保存一次（断点续传的关键）
        save_results(all_results)
        print(f"  已保存 {img_name}\n")
    
    print(f"全部完成！结果保存在 {OUTPUT_FILE}")
    print(f"缓存目录: {CACHE_DIR} （可删除以重新生成标注）")

if __name__ == "__main__":
    main()