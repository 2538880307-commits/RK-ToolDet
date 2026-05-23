# YOLO11n 工具检测 → RK3588 NPU 部署

基于 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 官方流程，将训练好的 YOLO11n 模型部署到 RK3588 NPU。

## 项目结构

```
├── best_tools.pt                   # 训练好的 PyTorch 模型
├── best_tools.onnx                 # 优化版 ONNX 模型（已导出 ✅）
├── dataset.txt                     # 量化校准图片列表
├── README.md
│
├── pt2onnx_rknn.py                 # 第一步：PT → 优化版 ONNX
├── gen_dataset_txt.py              # 生成校准图片列表
├── verify_onnx.py                  # 验证 ONNX 模型结构
│
├── ultralytics_yolo11/             # airockchip/ultralytics_yolo11 分支
│   └── ...
│
└── rknn_model_zoo/                 # 转换 & 推理脚本（基于官方 model_zoo）
    ├── py_utils/                   # 工具库
    │   ├── coco_utils.py
    │   ├── rknn_executor.py
    │   ├── onnx_executor.py
    │   └── pytorch_executor.py
    └── examples/yolo11/python/
        ├── convert.py              # 第二步：ONNX → RKNN（Linux）
        ├── yolo11.py               # 第三步：NPU 推理部署
        └── dataset.txt             # 量化图片列表（副本）
```

## 模型信息

| 项目 | 值 |
|---|---|
| 模型 | YOLO11n |
| 输入 | `[1, 3, 640, 640]` FP32 |
| 输出 | 9 个分离特征头（3 尺度 × 3 类输出） |
| 类别 | 25 类工具 |
| 优化 | DFL 移至 CPU、移除后处理结构、添加置信度总和分支 |

## 类别列表

```
0:活口扳手  1:示波器    2:油壶      3:铲子      4:塞尺
5:手摇把    6:加封钳子  7:医用钳    8:转辙机钥匙 9:插片
10:扭力扳手 11:扭力扳手盒 12:测距仪  13:手电筒   14:两用扳手
15:锤子     16:尖嘴钳   17:套筒     18:卷尺     19:老虎钳
20:螺丝刀   21:斜口钳   22:禁动牌   23:胶带     24:笔
```

## 三步部署流程

### 第一步：PT → 优化版 ONNX（Windows / Linux）

使用 airockchip 优化分支导出，移除 DFL、分离输出头：

```bash
python pt2onnx_rknn.py
```

输出：`best_tools.onnx`（覆盖标准导出）

### 第二步：ONNX → RKNN（仅 Linux）

> `rknn-toolkit2` 仅支持 Linux x86_64 / RK3588 板端

```bash
# 1. 安装 rknn-toolkit2
pip install rknn-toolkit2
# 或从 https://github.com/airockchip/rknn-toolkit2/releases 下载 .whl

# 2. 修改校准图片路径为 Linux 格式
#    dataset.txt 在 rknn_model_zoo/examples/yolo11/python/ 下
sed -i 's|E:/pycharm_py|/你的图片目录|g' rknn_model_zoo/examples/yolo11/python/dataset.txt

# 3. 执行转换
cd rknn_model_zoo/examples/yolo11/python
python convert.py ../../../../best_tools.onnx rk3588
```

输出：`rknn_model_zoo/examples/yolo11/python/best_tools.rknn`

### 第三步：RK3588 板端推理

```bash
# 安装运行时
pip install rknn-toolkit2-lite opencv-python numpy torch

# 单张图片推理
cd rknn_model_zoo/examples/yolo11/python
python yolo11.py --model_path best_tools.rknn --target rk3588 --img_folder /path/to/images --img_save

# 结果保存在 ./result/ 目录
```

## 关键依赖

| 阶段 | 依赖 | 平台 |
|---|---|---|
| PT→ONNX | ultralytics_yolo11, torch | Windows / Linux |
| ONNX→RKNN | rknn-toolkit2 (≥2.0) | Linux x86_64 / RK3588 |
| 板端推理 | rknn-toolkit2-lite, opencv-python, torch | RK3588 |

## 与标准导出的区别

| | 标准 ultralytics 导出 | 优化版（本项目） |
|---|---|---|
| 输出 | 单个 `[1, 29, 8400]` | 9 个分离头 |
| DFL | 在模型内 | 移至 CPU 后处理 |
| 量化友好 | ❌ | ✅ |
| rknn_model_zoo 兼容 | ❌ | ✅ |
| 导出脚本 | `model.export(format='onnx')` | `model.export(format='rknn')` |

## 参考

- [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- [ultralytics_yolo11 (RK 优化分支)](https://github.com/airockchip/ultralytics_yolo11)
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)