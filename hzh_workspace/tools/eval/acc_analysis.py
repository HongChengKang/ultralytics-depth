# -*- coding: utf-8 -*-
"""构建 int8 后做逐层量化误差分析（x86 模拟，snapshot 对比 fp32 vs quant）。"""
import os, sys
from rknn.api import RKNN

onnx = sys.argv[1]
dataset = sys.argv[2]
outdir = sys.argv[3]
imgs = sys.argv[4:]

rk = RKNN(verbose=False)
rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
assert rk.load_onnx(model=onnx) == 0
assert rk.build(do_quantization=True, dataset=dataset) == 0
os.makedirs(outdir, exist_ok=True)
assert rk.accuracy_analysis(inputs=imgs, output_dir=outdir) == 0
rk.release()
print("snapshot done:", outdir, os.listdir(outdir)[:10])
