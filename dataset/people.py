import os
import glob

def process_labels_and_images(labels_dir="labels", images_dir="images"):
    """
    处理标签和图片：
    1. 若标签文件中没有 class_id = 0 的行，则删除对应的 .jpg 图片和该 .txt 文件。
    2. 若有 class_id = 0 的行，则仅保留这些行，并删除第一列（类别列），
       其余边界框坐标保持不变，最后覆盖写入原 .txt 文件。
    """
    # 获取所有标签文件
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for txt_path in txt_files:
        # 获取对应的图片路径（假设图片扩展名为 .jpg）
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path = os.path.join(images_dir, base_name + ".jpg")
        
        # 读取并解析标签文件
        has_zero = False
        zero_lines = []  # 存储保留的 class_id=0 的行（去掉第一列）
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"警告：格式不正确，跳过该行：{line}")
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    print(f"警告：类别 ID 不是数字，跳过该行：{line}")
                    continue
                
                if class_id == 0:
                    has_zero = True
                    # 保留后四个坐标，重新组合成字符串
                    zero_lines.append(" ".join(parts[1:5]) + "\n")
        
        if not has_zero:
            # 删除对应图片（若存在）
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"已删除图片：{img_path}")
            else:
                print(f"图片不存在，跳过删除：{img_path}")
            # 删除标签文件
            os.remove(txt_path)
            print(f"已删除标签文件：{txt_path}")
        else:
            # 有 0 类别的框，覆盖写入处理后的内容
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(zero_lines)
            print(f"已更新标签文件：{txt_path}，保留了 {len(zero_lines)} 个类别为 0 的边界框")

if __name__ == "__main__":
    # 如果标签和图片目录不在当前目录，可以修改此处路径
    process_labels_and_images(labels_dir="D:\\DATASET\\BUU-SARD\\BUU-SARD\\labels\\test", images_dir="D:\\DATASET\\BUU-SARD\\BUU-SARD\\images\\test")
    print("处理完成！")