# hzh_workspace：DetDepth 数据 / 训练 / 导出 / 量化工作区

这里是「伪深度数据准备 → 检测+深度训练 → ONNX → RKNN → 板端验证」的整条链路。
板端评测工程在 3588 的 `~/hzh_ws/yolo-det`，两边通过 `weights/`、`models/` 里的文件对接。

**只允许改本目录**：`/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace`。

更细的目录/训练血缘分析见 [`WORKSPACE_ANALYSIS.md`](WORKSPACE_ANALYSIS.md)。

---

## 1. 当前目录结构

```
hzh_workspace/
├── train_coco_pf.sh           # ★ COCO_PF 正式配方（检测 yaml + 深度头 teacher 迁移）
├── train_pfv_dataset.sh       #   PFV_dataset 正式配方
├── train_coco_pf_scratch.sh   #   COCO_PF 纯从零（depth_weights=None，只做对照/冒烟）
├── gen_depth.sh               # ★ 伪深度生成入口（可重复给数据集补 depth）
├── env/                       # 运行环境（torch2.4.1+cu121 / ultralytics / onnx / cv2）
│   ├── pylibs/
│   └── cfg/                   # YOLO_CONFIG_DIR / MPLCONFIGDIR / TORCH_HOME
├── configs/                   # 训练 yaml（detdepth_from_scratch / pfv_detdepth / coco_pf_detdepth）
├── tools/                     # 训练 / 导出 / 转换 / 评测 / 伪深度 / split 脚本
│   ├── train/                 #   train_detdepth_{pf,pfv,from_scratch}.py
│   ├── data/                  #   gen_pseudo_depth.py
│   ├── export/                #   pt→onnx（单头 / 任意尺寸 / batch）
│   ├── convert/               #   onnx→rknn（fp16 / int8 / 混合精度）
│   ├── eval/                  #   onnxruntime mAP / 逐层量化误差
│   ├── predict/               #   PT 推理出 preds 文本
│   └── split_pipeline/        #   拆头导出 + RKNN 转换/推理一条龙
├── runs/                      # 训练输出（3 个正式 run）
├── logs/                      # 训练/导出日志（历史审计证据）
├── depth_gen/                 # 伪深度生成记录（报告/日志/对照图/校验）
├── models/                    # 导出的 RKNN（fp16 / hybrid / res1280 / misc / split_*）
├── weights/                   # 源 ONNX + 校准清单 + 早期 int8
└── datasets/                  # 板端一致性评测子集（不参与训练）
```

`archive/` 已于 2026-10-28 删除；`calib/`、`reports/`、`plan_v2/`、`kitti_joint_val/` 此前已清理；
工作区 `model_common/`、`runs/_aborted_foreground/`、4 个 split `.calib/` 也于 2026-10-28 删除。
历史结论保留在本 README 与 `WORKSPACE_ANALYSIS.md`。

---

## 2. 三个训练入口（按数据集命名）

| 脚本 | 数据集 | 配方 | 默认 run 目录 | 已知结果（best） |
|---|---|---|---|---|
| `train_coco_pf.sh` | `/mnt/data2/COCO_PF`（v1 2388/472 → v2 3188/672） | 检测从 `configs/detdepth_from_scratch.yaml` 起；深度头从仓库根 `model_common/yolo26n-depth.pt` 迁移 | `runs/detdepth_coco_pf`（v2 正式；v1 旧结果在 `detdepth_coco_pf_old`） | v1：136 ep，mAP50 **0.75214** / mAP50-95 **0.48695**；v2：151 ep，**0.79587 / 0.52981** |
| `train_pfv_dataset.sh` | `/mnt/data2/PFV_dataset`（8559/945） | 同上，数据换 PFV | `runs/detdepth_pfv_dataset` | 243 ep，**0.75458 / 0.45591** |
| `train_coco_pf_scratch.sh` | `/mnt/data2/COCO_PF` | `depth_weights=None`，检测+深度都从零 | `runs/detdepth_coco_pf_scratch` | 只做过冒烟，不是 v1/v2 配方 |

`train_coco_pf.sh` 支持前台/后台：

```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
./train_coco_pf.sh                  # 前台，400 epochs
./train_coco_pf.sh epochs=2         # 任意 ultralytics 覆盖
./train_coco_pf.sh start            # 后台（setsid，关 SSH 不停）
./train_coco_pf.sh status           # 进程 / GPU / results.csv 最后一行
./train_coco_pf.sh tail
./train_coco_pf.sh stop
```

后台记账仍是 `logs/pf_train.pid`、`logs/pf_train_latest.log`、`logs/pf_train_<时间>.log`；
`status` 默认看最新的 `detdepth_coco_pf` 或 `-N`（排除 `_old`），可用 `RUNDIR=... ./train_coco_pf.sh status` 指定别的 run。
当前 `pf_train_latest.log` 仍指向 v1 的历史日志（v2 没有保留全量日志）；未运行时 `status` 只把它标成 `latest artifact` 且不打印日志末尾，训练状态以 `results.csv` 为准；`tail` 看的也是这份历史日志，不要当成 v2。

---

## 3. 伪深度生成（`gen_depth.sh`）

给数据集补 `<root>/depth/<split>/<stem>.png`，16-bit PNG、单位毫米、与 RGB 同分辨率同名。
已存在的文件会跳过，重复跑只补新增图片。

```bash
# 最常用：COCO_PF 的 train/val
./gen_depth.sh --root /mnt/data2/COCO_PF --splits train val

# 只补 val；换 batch/device
./gen_depth.sh --root /mnt/data2/COCO_PF --splits val --batch 16 --device 0

# 换 teacher；默认是仓库根 model_common/yolo26n-depth.pt
./gen_depth.sh --root /mnt/data2/COCO_PF --splits train \
  --weights /path/to/teacher.pt
```

`--root` 必填；teacher 默认路径由脚本显式补上，不依赖 `tools/data/gen_pseudo_depth.py` 的内部默认值。
历史记录和两次增量报告在 [`depth_gen/`](depth_gen/README.md)。

---

## 4. 训练产物

`runs/` 下三个正式 run：

| run | 时间 | epochs | best mAP50 / mAP50-95 | 说明 |
|---|---|---:|---|---|
| `detdepth_pfv_dataset` | 10-23 12:08–22:16 | 243 | 0.75458 / 0.45591 | PFV_dataset |
| `detdepth_coco_pf_old` | 10-27 13:57–14:53 | 136 | 0.75214 / 0.48695 | COCO_PF v1（2388/472），旧结果 |
| `detdepth_coco_pf` | 10-27 18:54–20:56 | 151 | 0.79587 / 0.52981 | COCO_PF v2（扩到 3188/672），当前正式 |

命名规范：`detdepth_<数据集>[_old|_scratch]`。
v1 原目录 `pf_detdepth_from_scratch` → `detdepth_coco_pf_old`；
v2 原目录 `pf_detdepth_from_scratch_v2` → `detdepth_coco_pf`；
PFV 原目录 `pfv_detdepth_from_scratch-3` → `detdepth_pfv_dataset`。
run 目录里的 `args.yaml` 仍保留训练时的原名与 `save_dir`，只作历史记录，不要据此判断当前路径。

原来的 `runs/_aborted_foreground/` 已在 2026-10-28 清理（67 MB）：10-27 正式 v1 之前三次启动尝试，
batch 256/128 双卡 OOM、batch 80 单卡跑 2 epoch 后停掉；过程说明备份在
[`logs/aborted_foreground_20261027.md`](logs/aborted_foreground_20261027.md)。

---

## 5. 导出与量化

```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
R=/home/kang/workspace/projects/yolo-depth/ultralytics-depth

# a) PT → ONNX（任意权重/尺寸；单头用 pt_to_onnx.py）
python3 tools/export/export_onnx_any.py <best.pt> <imgsz> <out.onnx>

# b) ONNX → RKNN fp16 / int8
python3 tools/convert/convert.py   weights/pfv3.onnx rk3588 fp16
python3 tools/convert/convert.py   weights/pfv3.onnx rk3588 i8

# c) 拆头导出 + RKNN（推荐 int8 路线）
bash tools/split_pipeline/run_rknn_pipeline.sh ...
# 默认 v2：runs/detdepth_coco_pf/weights/best.pt -> models/split_pf_v2
# 产物前缀随 run 目录名，新文件为 detdepth_coco_pf_split_*；目录里已有的旧文件仍是 pf_detdepth_from_scratch_v2_*
# v1 旧模型：显式传 runs/detdepth_coco_pf_old/weights/best.pt 和 models/split_pf

# d) ONNX 侧指标
python3 tools/eval/eval_onnx.py <onnx> <images_dir> <out.txt> [conf] [nms]
```

- 新 ONNX → `weights/`；新 RKNN → `models/{fp16,hybrid,res1280,misc}/`；
- split 产物 → `models/split_pf*/`；4 个 `.calib/` 校准缓存已删，重跑 split 量化会重建；
- 校准图清单：`weights/dataset.txt`（300 行，来自 `/mnt/data2/VisDrone数据集/images/test`）。

---

## 6. 关键结论（已实测，别重复踩）

1. **PT→ONNX 无损、ONNX→RKNN(fp16) 无损**：同一套指标下 PT 0.4454 / ONNX 0.4454 / RKNN fp16 0.4459。
2. **单头 + int8 PTQ 会崩**：0.05 级别。`reg_max=1` 直接回归绝对坐标时，头部 `Sub/Add/Div/Concat`
   每一步 int8 requant 同向累积，框整体偏 +4px；GT 中位只有 10×20px，TP 腰斩。
3. **抢救办法（按性价比）**：① 拆多头导出（`export_onnx_split.py`，输出 raw `l,t,r,b`，解码移到图外 float）
   → int8 0.4375，且更快；② 混合精度点坐标链（11 个算子 fp16）→ 0.3148；③ 全 fp16 → 0.4459。
4. **calib 预 letterbox、换 1280、换量化算法都救不了单头 int8**；1280 只有 fp16 有效（0.5211）。
5. **teacher 路径**：训练和伪深度都用仓库根 `model_common/yolo26n-depth.pt`
   （12,998,295 B，md5 `4a4d810a0a783f68f9a3760f36c117be`）。工作区 `model_common/` 已于
   2026-10-28 删除（原 `.part` 残片），**不要再用工作区那个路径**。

---

## 7. 维护约定

- 新 ONNX → `weights/`；新 RKNN → `models/{fp16,hybrid,res1280,misc}/`；
  新 split 产物 → `models/split_pf*/`；报告/记录 → 对应目录的 `README.md` 或 `depth_gen/`。
- `tools/` 只放能复用的脚本；一次性实验脚本跑完就删（先确认不依赖它的产物）。
- `runs/` 三个正式 run、`models/` 交付物、`env/` 不要动；删任何东西前先确认没有引用。
- 训练日志留在 `logs/`，别混进根目录。
- 目录/训练血缘的详细分析见 [`WORKSPACE_ANALYSIS.md`](WORKSPACE_ANALYSIS.md)。
