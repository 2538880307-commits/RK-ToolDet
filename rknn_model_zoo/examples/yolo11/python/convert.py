import sys
import os

# 调整数据集路径
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.txt')
DEFAULT_RKNN_PATH = os.path.join(os.path.dirname(__file__), 'best_tools.rknn')
DEFAULT_QUANT = True


def parse_arg():
    if len(sys.argv) < 3:
        print("用法: python {} <onnx模型路径> <目标平台> [dtype(可选)] [输出路径(可选)]".format(
            sys.argv[0]))
        print("       目标平台: rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808")
        print("       dtype: i8/u8 (量化) 或 fp (不量化), 默认 i8")
        print("\n示例: python {} best_tools.onnx rk3588".format(sys.argv[0]))
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("错误: 无效的 dtype: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    if not os.path.exists(DATASET_PATH):
        print("错误: 找不到校准图片文件 '{}'".format(DATASET_PATH))
        print("请先运行 gen_dataset_txt.py 生成，或手动创建")
        exit(1)

    # 导入 RKNN (仅 Linux 可用)
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    print('--> 配置模型')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
    )
    print('完成')

    print('--> 加载 ONNX 模型')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('加载模型失败!')
        exit(ret)
    print('完成')

    print('--> 构建 RKNN 模型')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('构建模型失败!')
        exit(ret)
    print('完成')

    print('--> 导出 RKNN 模型')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('导出模型失败!')
        exit(ret)
    print('完成')

    rknn.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print('\n✅ RKNN 模型已导出: {} ({:.2f} MB)'.format(output_path, size_mb))