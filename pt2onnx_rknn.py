import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics_yolo11'))

from ultralytics import YOLO

model = YOLO(os.path.join(os.path.dirname(__file__), 'best_tools.pt'))
model.export(format='rknn')

print("\n✅ 优化版 ONNX 导出完成!")
print("   输出文件: best_tools.onnx (已覆盖为标准导出时的旧文件)")
print("   此 ONNX 已针对 RKNN 优化: 移除 DFL、分离多尺度输出头、添加置信度总和")