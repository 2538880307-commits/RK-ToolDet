import os
import glob

IMG_DIR = r"E:\pycharm_py\ultralytics-main\datasets\tools\images\val"
OUTPUT = "dataset.txt"

images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    images.extend(glob.glob(os.path.join(IMG_DIR, ext)))

with open(OUTPUT, "w", encoding="utf-8") as f:
    for img_path in images:
        f.write(img_path + "\n")

print(f"生成了 {len(images)} 条校准图片路径 -> {OUTPUT}")
print(f"⚠️ 在 Linux 上使用时，请将路径替换为 Linux 格式的路径")
print(f"   例如: sed -i 's|E:/pycharm_py/ultralytics-main|/home/user|g' dataset.txt")
print(f"   或者: sed -i 's|E:\\\\pycharm_py\\\\ultralytics-main|/home/user|g' dataset.txt")