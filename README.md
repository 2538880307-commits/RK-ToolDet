# RK-ToolDet — YOLO11n 工具检测 RK3588 NPU 部署

基于 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 官方流程，将训练好的 YOLO11n 模型部署到 RK3588 NPU 进行加速推理。

## 项目结构

```
├── best_tools.pt                   # 训练好的 PyTorch 模型
├── best_tools.onnx                 # 优化版 ONNX 模型（已导出）
├── dataset.txt                     # 量化校准图片列表（相对路径）
├── .gitignore
├── README.md
│
├── calibration/                    # 校准图片（82 张，25 类全覆盖）
│   ├── *.jpg
│   └── *.png
│
├── pt2onnx_rknn.py                 # 第一步：PT → 优化版 ONNX
├── pack_calibration.py             # 校准图片打包脚本
├── gen_dataset_txt.py              # 生成校准图片列表
├── verify_onnx.py                  # 验证 ONNX 模型结构
│
└── rknn_model_zoo/                 # 转换 & 推理脚本（基于官方 model_zoo）
    ├── py_utils/                   # 工具库
    │   ├── coco_utils.py
    │   ├── rknn_executor.py
    │   ├── onnx_executor.py
    │   └── pytorch_executor.py
    └── examples/yolo11/python/
        ├── convert.py              # 第二步：ONNX → RKNN（Linux）
        ├── yolo11.py               # 第三步：板端 NPU 推理（25 类已配置）
        └── dataset.txt
```

## 模型信息

| 项目 | 值 |
|---|---|
| 模型 | YOLO11n |
| 输入 | `[1, 3, 640, 640]` FP32 |
| 输出 | 9 个分离特征头（3 尺度 × 3 类输出） |
| 类别 | 25 类工具 |
| 优化 | DFL 移至 CPU、移除后处理结构、添加置信度总和分支 |
| ONNX 大小 | ~10 MB |

## 类别列表

```
0:活口扳手  1:示波器    2:油壶      3:铲子      4:塞尺
5:手摇把    6:加封钳子  7:医用钳    8:转辙机钥匙 9:插片
10:扭力扳手 11:扭力扳手盒 12:测距仪  13:手电筒   14:两用扳手
15:锤子     16:尖嘴钳   17:套筒     18:卷尺     19:老虎钳
20:螺丝刀   21:斜口钳   22:禁动牌   23:胶带     24:笔
```

## 快速开始

### 环境准备

```bash
git clone https://github.com/2538880307-commits/RK-ToolDet.git
cd RK-ToolDet
```

### 第一步：PT → 优化版 ONNX（Windows / Linux，已完成则跳过）

需先克隆 airockchip 优化分支：

```bash
git clone https://github.com/airockchip/ultralytics_yolo11.git --depth 1
python pt2onnx_rknn.py
```

输出：`best_tools.onnx`（已包含在仓库中，可直接用）

### 第二步：ONNX → RKNN（仅 Linux / RK3588 板端）

> `rknn-toolkit2` 仅支持 Linux，推荐直接在 RK3588 板上操作。

```bash
# 1. 安装 rknn-toolkit2
pip install rknn-toolkit2
# 或从 https://github.com/airockchip/rknn-toolkit2/releases 下载 .whl

# 2. 无需修改路径 —— dataset.txt 已用相对路径指向 calibration/
#    直接执行转换即可
cd rknn_model_zoo/examples/yolo11/python
python convert.py ../../../../best_tools.onnx rk3588
```

输出：`best_tools.rknn`

### 第三步：板端 NPU 推理

```bash
# 安装推理运行时
pip install rknn-toolkit2-lite opencv-python numpy torch

# 单张图片推理
cd rknn_model_zoo/examples/yolo11/python
python yolo11.py --model_path best_tools.rknn --target rk3588 --img_folder /path/to/images --img_save

# 结果保存在 ./result/ 目录
```

## 关键依赖

| 阶段 | 依赖 | 平台 |
|---|---|---|
| PT → ONNX | ultralytics_yolo11, torch | Windows / Linux |
| ONNX → RKNN | rknn-toolkit2 (≥2.0) | Linux / RK3588 |
| 板端推理 | rknn-toolkit2-lite, opencv-python, numpy, torch | RK3588 |

## 与标准导出的区别

| | 标准 ultralytics 导出 | RK-ToolDet（本项目） |
|---|---|---|
| 输出 | 单个 `[1, 29, 8400]` | 9 个分离头 |
| DFL | 在模型内 | 移至 CPU 后处理 |
| 量化友好 | ❌ | ✅ |
| rknn_model_zoo 兼容 | ❌ 不兼容 | ✅ 完美兼容 |
| 校准图片 | 需手动收集 | 82 张已打包，25 类全覆盖 |

## 参考

- [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- [ultralytics_yolo11 (RK 优化分支)](https://github.com/airockchip/ultralytics_yolo11)
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)