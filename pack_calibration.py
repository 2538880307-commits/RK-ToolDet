import os
import shutil

DEST_DIR = os.path.join(os.path.dirname(__file__), 'calibration')
os.makedirs(DEST_DIR, exist_ok=True)

DATASET_TXT = os.path.join(os.path.dirname(__file__), 'dataset.txt')
DATASET_TXT_ZOO = os.path.join(os.path.dirname(__file__), 'rknn_model_zoo', 'examples', 'yolo11', 'python', 'dataset.txt')

copied = []
with open(DATASET_TXT, 'r', encoding='utf-8') as f:
    for line in f:
        src = line.strip()
        if not src:
            continue
        fname = os.path.basename(src)
        dst = os.path.join(DEST_DIR, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied.append(dst)

print(f"复制完成: {len(copied)} 张图片 -> {DEST_DIR}")

# 更新 dataset.txt 为相对路径
relative_lines = [os.path.join('calibration', os.path.basename(p)) + '\n' for p in copied]
for txt_path in [DATASET_TXT, DATASET_TXT_ZOO]:
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(relative_lines)
    print(f"已更新: {txt_path} ({len(relative_lines)} 条, 相对路径)")