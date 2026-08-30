import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# ---------- 配置参数 ----------
JSON_FILE = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\1.json"                # 输入的JSON文件路径
OUTPUT_DIR = "D:\\DATASET\\BUU-SARD\\BUU-SARD\\labels\\xml"           # 输出XML文件的目录
IMAGE_WIDTH = 640                   # 默认图片宽度（像素）
IMAGE_HEIGHT = 640                  # 默认图片高度（像素）
IMAGE_DEPTH = 3                     # 默认图片通道数（RGB=3）

# ---------- 函数定义 ----------
def create_xml_for_image(filename, objects, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=IMAGE_DEPTH):
    """
    为单张图片创建XML树结构
    """
    root = ET.Element("annotation")

    # filename
    ET.SubElement(root, "filename").text = filename

    # size
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    # 遍历每个目标（objects 是字典，key为"0","1",...）
    for obj_id, obj_data in objects.items():
        obj = ET.SubElement(root, "object")

        # pose
        pose = obj_data.get("pose", "Others")
        ET.SubElement(obj, "pose").text = pose

        # bndbox
        bndbox = obj_data.get("bndbox", {})
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(bndbox.get("xmin", 0))
        ET.SubElement(box, "ymin").text = str(bndbox.get("ymin", 0))
        ET.SubElement(box, "xmax").text = str(bndbox.get("xmax", 0))
        ET.SubElement(box, "ymax").text = str(bndbox.get("ymax", 0))

        # injured
        injured = obj_data.get("injured", 0)
        ET.SubElement(obj, "injured").text = str(injured)

        # shirt_color
        shirt_color = obj_data.get("shirt_color", "unknown")
        ET.SubElement(obj, "shirt_color").text = shirt_color

    return root

def save_xml(root, output_path):
    """
    将XML树写入文件，并自动添加缩进
    """
    # 将ElementTree转换为字符串
    rough_string = ET.tostring(root, encoding='utf-8')
    # 使用minidom美化（添加缩进）
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # 写入文件（去除多余的换行符，但保留结构）
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

def main():
    # 检查JSON文件是否存在
    if not os.path.exists(JSON_FILE):
        print(f"错误：文件 {JSON_FILE} 不存在")
        return

    # 读取JSON
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 遍历每张图片
    for filename, objects in data.items():
        if not objects:  # 跳过空标注
            continue

        # 构建XML
        root = create_xml_for_image(filename, objects)

        # 生成输出文件名（与图片同名，扩展名为.xml）
        xml_filename = filename.replace('.jpg', '.xml') if filename.endswith('.jpg') else filename + '.xml'
        output_path = os.path.join(OUTPUT_DIR, xml_filename)

        # 保存XML
        save_xml(root, output_path)
        print(f"已生成：{output_path}")

    print("全部转换完成！")

if __name__ == "__main__":
    main()