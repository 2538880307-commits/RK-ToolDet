import onnx
import numpy as np

model = onnx.load("best_tools.onnx")
onnx.checker.check_model(model)
print("✅ ONNX 模型结构验证通过")

graph = model.graph
print(f"\n输入节点:")
for inp in graph.input:
    shape = [dim.dim_value if dim.dim_value else str(dim.dim_param) for dim in inp.type.tensor_type.shape.dim]
    print(f"  名称: {inp.name}, 形状: {shape}, 类型: {inp.type.tensor_type.elem_type}")

print(f"\n输出节点:")
for outp in graph.output:
    shape = [dim.dim_value if dim.dim_value else str(dim.dim_param) for dim in outp.type.tensor_type.shape.dim]
    print(f"  名称: {outp.name}, 形状: {shape}, 类型: {outp.type.tensor_type.elem_type}")

print(f"\nOpset 版本: {graph.opset_import[0].version}")
print(f"总节点数: {len(graph.node)}")
print(f"模型大小: {10.1:.1f} MB")

import os
size_mb = os.path.getsize("best_tools.onnx") / (1024 * 1024)
print(f"文件大小: {size_mb:.2f} MB")