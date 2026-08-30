import os
import cv2
import csv
from pathlib import Path
from tqdm import tqdm

def compute_sequence_sizes(seq_root, output_csv=None):
    """
    遍历 seq_root 下的每个序列，计算：
    - 原始图片文件总大小 (MB)
    - 重新编码为 JPEG 质量 85 后的总大小 (MB)
    - 比率 (上传/原始)
    结果保存为 CSV，并打印表格。
    """
    seq_root = Path(seq_root)
    results = []
    total_orig_all = 0.0
    total_enc_all = 0.0

    # 获取所有包含 color 目录的序列
    seq_dirs = [p for p in seq_root.iterdir() if p.is_dir() and (p / "color").exists()]
    seq_dirs.sort()

    for seq_path in tqdm(seq_dirs, desc="Processing sequences"):
        color_dir = seq_path / "color"
        img_files = sorted([f for f in color_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not img_files:
            continue

        total_orig_bytes = 0
        total_enc_bytes = 0

        for img_path in img_files:
            # 原始文件大小
            orig_size = img_path.stat().st_size
            total_orig_bytes += orig_size

            # 读取图像并重新编码为 JPEG 质量 85
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            enc_size = len(encoded.tobytes())
            total_enc_bytes += enc_size

        # 转换为 MB
        orig_mb = total_orig_bytes / (1024 * 1024)
        enc_mb = total_enc_bytes / (1024 * 1024)
        ratio = enc_mb / orig_mb if orig_mb > 0 else 0

        results.append({
            'seq_name': seq_path.name,
            'orig_mb': orig_mb,
            'enc_mb': enc_mb,
            'ratio': ratio
        })

        total_orig_all += orig_mb
        total_enc_all += enc_mb

    # 计算总体平均比率
    avg_ratio = total_enc_all / total_orig_all if total_orig_all > 0 else 0

    # 打印表格
    print("\n{:<15} {:>12} {:>12} {:>10}".format("Sequence", "Orig (MB)", "Enc (MB)", "Ratio"))
    print("-" * 52)
    for r in results:
        print("{:<15} {:>12.4f} {:>12.4f} {:>10.4f}".format(
            r['seq_name'], r['orig_mb'], r['enc_mb'], r['ratio']))
    print("-" * 52)
    print("{:<15} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        "TOTAL/AVG", total_orig_all, total_enc_all, avg_ratio))

    # 如果指定了输出 CSV，写入文件
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['seq_name', 'orig_mb', 'enc_mb', 'ratio'])
            writer.writeheader()
            writer.writerows(results)
            # 添加汇总行
            writer.writerow({'seq_name': 'TOTAL/AVG', 'orig_mb': total_orig_all,
                             'enc_mb': total_enc_all, 'ratio': avg_ratio})
        print(f"\n结果已保存至: {output_csv}")

    return results, avg_ratio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare original vs JPEG85 encoded sizes of UAV123 sequences.")
    parser.add_argument("--data_root", type=str, default="/workspace/UAV123/UAV123/sequences",
                        help="Root directory of UAV123 dataset")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional CSV output file")
    args = parser.parse_args()

    compute_sequence_sizes(args.data_root, args.output_csv)