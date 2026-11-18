# hzh_workspace：数据 / 训练 / 导出 / 量化工作区

从数据到板端的完整链路，两条技术路线并存：

```
数据与标签 (5 列检测 / 6 列检测+距离)
   → 训练（metdist：检测框 + 框级距离回归   /   detdepth：检测框 + 稀疏深度图）
   → .pt → ONNX（拆头多输出）→ RKNN（fp16 / int8）
   → RK3576 / RK3588 板端（工程在板端，本工作区只负责模型产物）
```

边界（重要）：

- **只允许改本目录**：`/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace`。
- 仓库源码 `ultralytics/` **只读**：所有改动靠"复制 + PYTHONPATH 注入"，不改仓库。
- 数据集 `/mnt/data2/**` **只读**（唯一例外是 detdepth 路线的 `depth/` 伪深度，由 `gen_depth.sh` 负责；标签目录只允许新增 `labels_dist*`）。
- 训练环境固定：`/home/kang/miniforge3/envs/yolo/bin/python` + `PYTHONPATH=<repo>` + `YOLO_CONFIG_DIR=<ws>/env/cfg`（site-packages 的 ultralytics 太旧，缺 Depth 头）。

---

## 1. 目录结构

```
hzh_workspace/
├── train_metdist.sh           # ★ metdist 正式入口（检测 + 框级距离，当前主力）
├── train_coco_pf.sh           # ★ detdepth 正式入口（检测 + 稀疏深度图头，COCO_PF）
├── gen_depth.sh               #   伪深度生成入口（detdepth 路线补 depth/<split>/*.png）
├── README.md
│
├── configs/
│   ├── metdist_dataset.yaml   # ★ metdist 数据集唯一事实来源（dataset_root 一行）
│   ├── metdist_data.yaml      #   （生成物）images 路径 + names，make_data_yaml.py 写
│   ├── metdist_smoke_data.yaml#   （生成物）20 train + 10 val 冒烟清单
│   ├── metdist_model.yaml     #   网络结构（YOLO26n + Detect(P3,P4,P5)，nc=2）
│   ├── metdist_train.yaml     #   薄覆盖文件（默认值在 train_metdist.py DEFAULTS）
│   ├── metdist_best.yaml      # ★ 当前最优配方（反例见文件头注释）
│   ├── detdepth_from_scratch.yaml / coco_pf_detdepth.yaml / pfv_detdepth.yaml
│
├── libs/
│   └── metdist/               # head.py data.py loss.py model.py trainer.py augment.py
│
├── tools/
│   ├── train/                 # train_metdist.py / train_detdepth_{pf,pfv,from_scratch}.py
│   ├── data/                  # calibrate_dist_labels.py  标签仿射标定
│   │                          # verify_cal_labels.py      标定结果独立校验
│   │                          # make_own_dist_labels.py   非自采域距离置 -1（掩码）
│   │                          # make_dist_labels.py       教师一步式 6 列标签
│   │                          # make_data_yaml.py         dataset_root → data yaml
│   │                          # gen_pseudo_depth.py       detdepth 伪深度 PNG
│   ├── eval/                  # eval_metdist.py           own 域距离评估（图 + CSV + summary）
│   │                          # acc_analysis.py           int8 逐层量化误差
│   ├── export/                # pt → onnx（单输出 / 任意尺寸 / 任意 batch）
│   ├── convert/               # onnx → rknn（fp16 / int8 / 混合精度）
│   ├── predict/               # PT 推理出 preds 文本（与板端 C++ 预处理对齐）
│   ├── vis/                   # vis_dist3.py
│   ├── split_pipeline/        # detdepth：拆头导出 + RKNN（10 输出）
│   └── split_pipeline_metdist/# metdist：拆头导出 + RKNN（12 输出，含 pipeline.env）
│
├── runs/                      # 训练输出（weights / results.csv / 曲线）
├── models/                    # 交付板端的 RKNN（按日期 + 芯片分目录）
├── datasets/                  # 评测/冒烟用的小清单（不参与正式训练）
├── depth_gen/                 # detdepth 伪深度生成的记录
├── docs/                      # metdist.md、detdist_board_handover.md（板端交接）
├── logs/                      # 训练/导出/转换/评估日志
└── env/                       # YOLO_CONFIG_DIR、MPLCONFIGDIR、TORCH_HOME、pylibs
```

---

## 2. 两条技术路线

| | **metdist**（当前主力） | **detdepth**（早期路线） |
|---|---|---|
| 目标 | 检测框 + 每框一个米制距离 | 检测框 + 稀疏深度图 |
| 距离表示 | 每级一个距离塔（3×3×2，宽 32），输出 **log 深度**，导出时 `exp(clamp(logit))` → 米 | Depth 头输出 `(H,W)` 深度图 |
| 读取规则 | 框内 anchor 中位数 <5 m 取 **p20**，否则中位数 | 框内深度中位数 |
| 标签 | 6 列 `cls cx cy w h d_norm`（`labels_dist*`） | 5 列检测 + `depth/<split>/*.png`（uint16 毫米） |
| 损失 | log 空间 Huber + 非对称 τ + 近场加权（+可选序关系） | 教师蒸馏深度损失 |
| 数据集 | `/mnt/data2/hzh/PF_11.9` | `/mnt/data2/COCO_PF`、`/mnt/data2/PFV_dataset` |
| ONNX 输出 | 拆头 12 输出（每级 box/cls/score_sum + 三级距离图） | 拆头 10 输出（9 + 深度图） |
| 入口 | `train_metdist.sh` | `train_coco_pf.sh` |
| 文档 | `docs/metdist.md` | 本 README §9 + `depth_gen/README.md` |

---

## 3. 数据集与标签

| 数据集 | 图 | 距离标签 | 说明 |
|---|---|---|---|
| `/mnt/data2/hzh/PF_11.9` | train 4034 / val 600 | `labels_dist/`（教师原始 6 列） | 教师标定前的基线标签 |
| 同上 | 同上 | `labels_dist_cal/`（own_video 仿射校正 a=1.38/b=0.37） | 校正后视为"真值尺度" |
| 同上 | 同上 | `labels_dist_own/`（coco/fpv 距离置 -1） | **距离监督只给自采域**；检测标签不变 |
| `/mnt/data2/COCO_PF` | 2088/397 | `depth/`（教师 PNG，1000=1 m） | detdepth |
| `/mnt/data2/PFV_dataset` | 8559/945 | 同上 | detdepth |

标签约定：

- metdist 只认 **6 列**，少列/多列直接报错；
- `d_norm = clip(d_m, 0, 15)/15`（6 位小数）；有效像素不足或整框 >15 m 写 `-1`；
- `-1` 只跳过距离损失，检测损失照常；
- 生成/校正/掩码三个脚本见 `tools/data/`，每次产出都配 `stats/` 下报告。

---

## 4. 训练入口

### 4.1 metdist（`train_metdist.sh`）

```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace

python $P/tools/data/make_data_yaml.py --smoke        # 数据集位置变了才需要
python $P/tools/train/train_metdist.py --dry-run      # 只打印生效参数

$P/train_metdist.sh start                             # 后台（PID → logs/metdist.pid）
$P/train_metdist.sh start --config configs/metdist_best.yaml   # ★ 最优配方
$P/train_metdist.sh tail | status | stop
DETDIST_TAG=exp1 $P/train_metdist.sh start --name exp1 # 并行实验用独立 pid/日志
```

- 三层覆盖：`tools/train/train_metdist.py` DEFAULTS < `configs/metdist_train.yaml` < CLI；
- 关键默认：`imgsz 640 / batch 96 / epochs 160 / SGD lr0 0.01 / cos_lr / patience 40 /
  dist_lambda 3.0 / dist_huber_delta 0.2 / dist_tau 0.5 / scale 0.1 / erasing 0.2`；
- **热启动**：`det_weights`（默认 `runs/detdist_pf_cal/weights/best.pt`）加载检测权重，距离塔新初始化；
  `depth_weights` 可选（深度 backbone，混装冒烟会 mAP 归零，默认关）；
- **best 选择**：fitness = mAP50-95 − 2×val/dist（距离感知；纯 mAP 会在 ep2 早停）；
- 数据：train 读 `labels_dist_own`（默认），val 也走 `DistDataset`（不写 `.cache`）；
- 环境变量（libs 层）：`DETDIST_V2_LAMBDA/HUBER_DELTA/TAU/NEAR_WEIGHT/RANK/METRIC_OWN/DROPOUT/BGSWAP/MID`。

### 4.2 detdepth（`train_coco_pf.sh`）

```bash
$P/train_coco_pf.sh start      # 后台，PID → logs/pf_train.pid
$P/train_coco_pf.sh status | tail | stop   # 不带子命令 = 前台
```

- 数据/模型：`configs/{coco_pf_detdepth,detdepth_from_scratch}.yaml`；
- 入口：`tools/train/train_detdepth_pf.py`（深度头从 `model_common/yolo26n-depth.pt` 迁移）；
- 产物：`runs/detdepth_coco_pf/`。

---

## 5. 结果在哪看

- `runs/<name>/results.csv`：逐 epoch `box/cls/dfl(/dist)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`；
- `runs/<name>/weights/{best,last}.pt`；
- `tools/eval/eval_metdist.py --weights runs/<name>/weights/last.pt` → `images/ + boxes.csv + summary.json`（距离分桶、偏置、ρ、逐框 ≤0.5 m 比例）。

---

## 6. metdist 实现要点

- **不改仓库**：yaml 仍写 `Detect`；`libs/metdist/model.py` 构造后把 head 类切成 `DetectDistV2`，
  每级挂一个距离塔（Conv3×3(ch,32)+Conv3×3(32,1)，+13 万参数，总 2.63 M）；
- **输出**：`preds["one2many"]["dist"]` 为 log 深度（anchor 顺序 P3→P5）；导出时 `exp(clamp(., log0.2, log15))`；
- **损失**：log 空间 Huber；τ<0.5 时近场高估重罚（读数偏小）；近场样本加权；可选图内序关系损失
  （coco/fpv 只参与排序、不参与米制回归，用 `metric_own` 控制）；可选背景置换增强与塔 dropout；
- **数据**：`DistDataset` 自定义 `get_labels` 读 6 列且不写 cache；几何增强时 dist 与框同步裁剪；
- **读取**：框内 anchor 中位数 <5 m 取 p20（近场偏保守），否则中位数；板端/脚本口径一致。

细节与回退见 [`docs/metdist.md`](docs/metdist.md)。

---

## 7. 导出 / 量化 / 板端

| 目标 | 脚本 | 输出 |
|---|---|---|
| .pt → ONNX（整图） | `tools/export/pt_to_onnx.py` 等 | 检测 + 距离拼接 |
| .pt → ONNX（拆头） | `tools/split_pipeline/export_onnx_split.py`（detdepth 10 输出）<br>`tools/split_pipeline_metdist/export_onnx_split.py`（metdist 12 输出） | 12 路：box/cls/score_sum ×3 + dist_p3/p4/p5 |
| ONNX → RKNN | `tools/convert/`、`tools/split_pipeline_metdist/convert_rknn.py` | fp16 / int8 |
| 一条龙 | `tools/split_pipeline_metdist/run_rknn_pipeline.sh` | `models/split_pf_<date>_<chip>/` + 预览 |

- 可调项集中在 `tools/split_pipeline_metdist/pipeline.env`（PT/OUT/PLATFORM/IMGSZ/CALIB_*/NEAR_PCT）；
- 产物命名：`<run名>_split_b<batch>_<chip>_{fp16,int8}.rknn`；
- **int8 注意**：`dist_p3/p4/p5` 可能输出整型，读数前按 `zp/scale` 反量化；fp16 无此问题；
- 当前交付：`models/split_pf_11_17_3576/`、`models/split_pf_11_17_3588/`（metdist 最优权重）。

板端实测（RK3588，int8，图在内存，p50）：

| 模型 | NPU | 端到端 |
|---|---|---|
| metdist（12 输出） | 40.7 ms | 43.6 ms |
| metdist fp16 | 128.9 ms | 131.3 ms |
| detdepth（10 输出，历史） | 114.5 ms | 176.2 ms |

---

## 8. 评估与可视化

- `tools/eval/eval_metdist.py`：own 域距离评估（`--logdepth/--pct/--near-pct`），标注图 + CSV + summary；
- `tools/eval/acc_analysis.py`：int8 逐层量化误差；
- `tools/vis/vis_dist3.py`：GT/预测双色框 + 距离标注。

---

## 9. 伪深度生成（detdepth 路线）

```bash
$P/gen_depth.sh --root /mnt/data2/COCO_PF --splits train val
```

- 教师 `model_common/yolo26n-depth.pt`，输出 `depth/<split>/*.png`（uint16 毫米）；
- 记录与坑见 `depth_gen/README.md`。

---

## 10. 日志与环境

- 日志：`logs/{metdist,pf_train,split_*}_<ts>.log` + `*_latest.log` 软链 + `*.pid`；
- 必需环境：
  ```bash
  PY=/home/kang/miniforge3/envs/yolo/bin/python
  export PYTHONPATH=<repo>:<repo>/hzh_workspace/libs
  export YOLO_CONFIG_DIR=<repo>/hzh_workspace/env/cfg
  ```
- `env/pylibs/` 为额外私有依赖（numpy 等）。

---

## 11. 维护约定

1. `tools/` 只放可复用脚本；一次性实验跑完即删（删前 `grep -rn` 确认无引用）；
2. `runs/`、`models/`、`env/` 不要动；删产物前确认无板端/文档引用；
3. 不往数据集写东西：metdist 不写 `.cache`；detdepth 走原生 `YOLODataset` 会在 `labels/` 下生成 `*.cache`，属工具行为，可事后删；
4. 链路命名：入口 `<name>.sh`、实现 `libs/<name>/`、配置 `configs/<name>_*.yaml`、日志 `logs/<name>_*.log`、文档 `docs/<name>.md`；
5. 改默认值 → 改 `tools/train/*.py` 的 `DEFAULTS`；改目录结构 → 同步本 README 与 `docs/`。
