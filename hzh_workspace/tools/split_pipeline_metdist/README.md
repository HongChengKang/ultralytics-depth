# split_pipeline_detdist — 支持 detdist（检测头距离回归）的 RKNN 部署流水线

这是 `tools/split_pipeline/` 的最小改动副本，只为让 detdist 的 `best.pt`（`DetectDist` 头）能走同一套
「.pt → split ONNX → fp16/int8 RKNN → ONNX 预览」流程。旧 detdepth 头仍然可用，脚本自动识别。

## 旧流程为什么不认 detdist（三处硬编码）
1. `export_onnx_split.py` 只把 `HERE/REPO/hck_workspace/detdepth` 加进 `sys.path`，而 detdist 的
   checkpoint pickle 引用 `detdist.head.DetectDist` → 反序列化就要 ImportError。
2. `split_forward` 末尾固定 `self.dhead(x)`，`DetectDist` 没有 `dhead` → AttributeError。
3. 产物固定 10 个输出（9 检测 + 1 张 depth 图）；detdist 是 12 个（9 检测 + P3/P4/P5 三张距离图，单位米），
   `rknn_infer` 里 `depth = outs[9]`、`convert_rknn` 里 `sim[9]` 的假设随之失效。

## 改动（都在本目录内，未动上一级原流程）
| 文件 | 改动 |
|---|---|
| `export_onnx_split.py` | `sys.path` 增加 `hzh_workspace/libs`；按 `hasattr(head,"dist")` 选两种 forward：`split_forward_dist` 输出 `dist_p3/p4/p5 = sigmoid(raw)*15`（米）；旧头仍走 `dhead` + `depth`。`--imgsz` 默认 `480x320`。 |
| `rknn_infer.py` | 支持 10/12 输出：`box_dist_levels()` 取框内 P3/P4/P5 anchor 的中位数（米）；对 int8 输出打印反量化警告；标签显示 `d=..m`。 |
| `convert_rknn.py` | 精度对比里第 10 个输出按 12 输出（dist_p3）解释；`map_diff` 除零保护。 |
| `probe_split_onnx.py` | 打印附加的距离图输出与 min/median/max。 |
| `run_rknn_pipeline.sh` | 工具目录改为"脚本自己所在目录"；`PT` 不存在时报错；`IMGSZ` 缺省从 `args.yaml` 读（支持 `[480, 320]`）。 |

## 用法（与旧流程一致）
```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
bash tools/split_pipeline_detdist/run_rknn_pipeline.sh \
  runs/dist_pf_all/weights/best.pt models/split_pf_11_10_3588
# 板子是 rk3576 时：PLATFORM=rk3576 bash tools/split_pipeline_detdist/run_rknn_pipeline.sh ...
```
单步：
```bash
PYTHONPATH=$PWD/libs:.. /home/kang/miniforge3/envs/yolo/bin/python \
  tools/split_pipeline_detdist/export_onnx_split.py runs/dist_pf_all/weights/best.pt /tmp/split.onnx --imgsz 480x320
/home/eav-nanjin/miniforge3/envs/rknn/bin/python \
  tools/split_pipeline_detdist/convert_rknn.py /tmp/split.onnx /tmp/split_fp16.rknn --platform rk3588 --fp
```

## 已验证（随机权重，非训练）
- 12 个输出形状正确：`box/cls/score_sum` ×3 + `dist_p3(60,40)/p4(30,20)/p5(15,10)`（480×320 输入）。
- 把 cls bias 置 0 后 `sigmoid=0.5` → 三张距离图恒为 **7.500 m**，证明导出的单位是米而非 logit。
- RKNN fp16 构建成功，模拟器输出与 ONNX 逐元素 **max|Δ|=0.0**；int8（64 张校准）也能构建。
- 精度对比里的 `matched 0` 是随机权重导致框退化（零面积）所致，真实 `best.pt` 才有意义。

## 注意
- `runs/dist_pf_all/weights/best.pt` 目前**不存在**（历史 run 已清），需要重训或指向别的 checkpoint；
  `run_rknn_pipeline.sh` 会对不存在的路径直接报错。
- int8 模型的 `dist_p*` 输出 dtype 会被改成 int8（构建日志有 warning），板端需用 zp/scale 反量化；
  fp16 无此问题。`rknn_infer.py` 遇到整型输出会打印提醒。
- 与旧流程同样的约束：`--imgsz` 必须等于训练尺寸，两边都是 32 的倍数；ONNX 是固定 batch。
