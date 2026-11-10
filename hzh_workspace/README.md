# hzh_workspace：数据 / 训练 / 导出 / 量化工作区

本目录是一条**从数据到板端**的完整链路：

```
数据与标签 (5 列检测 / 6 列检测+距离)
   → 训练（两种头：detdepth 稀疏深度图头 / detdist 框级距离回归头）
   → .pt
   → ONNX（整图或拆头多输出）
   → RKNN（fp16 / int8）
   → 3588 板端验证（板端工程在 3588 的 ~/hzh_ws/yolo-det，通过 weights/、models/ 对接）
```

边界（重要）：

- **只允许改本目录**：`/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace`。
- 仓库源码 `/home/kang/workspace/projects/yolo-depth/ultralytics-depth/ultralytics/` **只读**，本目录所有改动都靠“复制 + 注入”，不改仓库。
- 数据集 `/mnt/data2/**` **只读**（唯一例外是历史上生成过的 `depth/` 伪深度，由 `gen_depth.sh` 负责）。
- 训练必须用 `/home/kang/miniforge3/envs/yolo/bin/python` + `PYTHONPATH=<repo>` + `YOLO_CONFIG_DIR=<本目录>/env/cfg`，否则会用到 site-packages 里的旧版 ultralytics（没有 Depth 头）。

---

## 1. 目录结构

```
hzh_workspace/
├── train_detdist.sh           # ★ detdist 正式配方（检测头 + 框级距离回归）
├── train_coco_pf.sh           # ★ detdepth 正式配方（检测头 + 稀疏深度图头，COCO_PF）
├── gen_depth.sh               #   伪深度生成入口（给数据集补 depth/<split>/*.png）
├── README.md                  #   本文件（链路总览）
│
├── configs/                   # 训练 / 数据 / 模型 yaml
│   ├── detdist_dataset.yaml   #   ★ detdist 唯一数据集事实来源（dataset_root 一行）
│   ├── detdist_data.yaml      #   （生成物）images 路径 + names，由 make_data_yaml.py 写
│   ├── detdist_smoke_data.yaml#   （生成物）20 train + 10 val 冒烟清单
│   ├── detdist_model.yaml     #   detdist 网络结构（YOLO26n + Detect(P3,P4,P5)，nc=2）
│   ├── detdist_train.yaml     #   detdist 训练覆盖项（默认值在 tools/train/train_detdist.py）
│   ├── detdepth_from_scratch.yaml  # detdepth 网络结构（yaml 内写 Depth 头）
│   ├── coco_pf_detdepth.yaml  #   COCO_PF 数据集（5 列 + depth/，depth_scale=1000）
│   └── pfv_detdepth.yaml      #   PFV_dataset 数据集
│
├── libs/                      # 可复用实现（注入式，不改仓库）
│   ├── detdist/               #   head.py data.py loss.py model.py trainer.py（当前主用）
│   └── dist_plan/             #   仅 re-export DetectDist/DistModel，给旧 pickle 兜底
│
├── tools/
│   ├── train/                 #   train_detdist.py / train_detdepth_{pf,pfv,from_scratch}.py
│   ├── data/                  #   make_dist_labels.py（一步式距离标签）
│   │                          #   make_data_yaml.py（dataset_root → data yaml）
│   │                          #   gen_pseudo_depth.py（教师 → uint16 深度 PNG）
│   ├── eval/                  #   eval_detdist.py（mAP + 距离分桶）
│   │                          #   eval_onnx.py / dist_from_onnx.py / dist_both.py
│   │                          #   acc_analysis.py（int8 逐层误差）
│   ├── export/                #   pt → onnx（单输出 / 任意尺寸 / 任意 batch）
│   ├── convert/               #   onnx → rknn（fp16 / int8 / 混合精度）
│   ├── predict/               #   PT 推理出 preds 文本（与板端 C++ 预处理对齐）
│   ├── vis/                   #   vis_dist3.py（GT/预测 + 距离标注可视化）
│   ├── split_pipeline/        #   detdepth 头：拆头导出 + RKNN 转换 + 板端预览（10 输出）
│   └── split_pipeline_detdist/ #  detdist 头：同一套流水线（12 输出，含 pipeline.env）
│
├── runs/                      # 训练输出（weights/ results.csv / 曲线 / dist_eval.json）
├── models/                    # 交付给板端的 RKNN（按 run + 芯片分目录）
├── weights/                   # 源 ONNX / 量化校准清单 / 早期 int8
├── datasets/                  # 评测与冒烟用的小清单/子集（不参与正式训练）
├── depth_gen/                 # 伪深度生成的记录（报告/日志/对照图/校验脚本）
├── docs/                      # 分链路说明（detdist.md）
├── logs/                      # 训练 / 导出 / 转换 / 评估日志（历史审计证据）
└── env/                       # 运行环境（pylibs / cfg：YOLO_CONFIG_DIR、MPLCONFIGDIR、TORCH_HOME）
```

---

## 2. 两条技术路线

| | **detdist**（当前主力） | **detdepth**（早期路线） |
|---|---|---|
| 目标 | 检测框 + 每框一个距离 | 检测框 + 稀疏深度图 |
| 距离表示 | 检测头每级 1 个通道，`sigmoid→[0,1]`，`d_m = d_norm × 15` | Depth 头输出 `(H,W)` 深度图（米） |
| 标签 | `labels_dist/*.txt`：6 列 `cls cx cy w h d_norm` | 5 列检测 + `depth/<split>/<stem>.png`（uint16 毫米） |
| 损失 | `box + cls + dfl + λ·Huber(dist)`，λ 默认 0.5 | `box + cls + dfl + λ·depth`（教师蒸馏） |
| 数据集 | `/mnt/data2/hzh/PF_11.9` | `/mnt/data2/COCO_PF`、`/mnt/data2/PFV_dataset` |
| ONNX 输出 | 整图 2 输出；拆头 12 输出（3 级 box/cls/score + 3 级 dist） | 整图 2 输出；拆头 10 输出（3 级 + 深度图） |
| 入口 | `train_detdist.sh` | `train_coco_pf.sh` |
| 说明文档 | [`docs/detdist.md`](docs/detdist.md) | 本 README 第 6 节 + `depth_gen/README.md` |

### 2.1 推理速度

PC 侧（RTX A6000，PT fp32，`640×640`，不含前后处理；warmup 后 30 次/20 次取均值）：

| 模型 | 参数量 | batch=1 | batch=16 |
|---|---|---|---|
| detdist（`runs/dist_pf_all`） | 2.51 M | 9.07 ms/图（110 图/s） | 938 图/s |
| detdepth（`runs/detdepth_coco_pf`） | 5.42 M | 9.52 ms/图（105 图/s） | 471 图/s |

板端 RK3588 实测（**int8 RKNN 模型**，不是 fp16、也不是 PC 上的 fp32；单位 ms；端到端 = 含前后处理，差值 = 端到端 − NPU）：

| 模型 | NPU | 端到端 | 差值 |
|---|---|---|---|
| 新（detdist，12 输出） | 47.0 | 111.9 | 64.9 |
| 旧（detdepth，10 输出） | 114.5 | 176.2 | 61.7 |

结论：同一 backbone、同为 int8 部署时，detdist 把 NPU 段压到旧方案的 41%，端到端约 1.6×；前后处理仍占 ~58%。
int8 下 `dist_p3/p4/p5` 可能输出整型，读数前要按 `zp/scale` 反量化（见第 7 节）。

---

## 3. 数据集与标签

| 数据集 | 图/标签 | 距离标签来源 | 备注 |
|---|---|---|---|
| `/mnt/data2/hzh/PF_11.9` | `images/{train,val}` 4034/600，`labels/{train,val}` 5 列 | `labels_dist/{train,val}` 6 列（4634 文件） | detdist 用；`d_norm=-1` 表示该框距离不可用（ignore），**检测标签照常参与** |
| `/mnt/data2/COCO_PF` | 2088/397，5 列 + `depth/` | 教师模型生成 `depth/<split>/*.png`（uint16，1000=1 m） | detdepth 用；`configs/coco_pf_detdepth.yaml` |
| `/mnt/data2/PFV_dataset` | 8559/945 | 同上 | `configs/pfv_detdepth.yaml` |

关于标签的约定：

- detdist 只认 **6 列**；少一列/多一列直接报错，不静默丢框。
- `d_norm = clip(d_m, 0, 15) / 15`（6 位小数）；有效像素不足 10 个或整框在 15 m 外时写 `-1`。
- 写 `-1` 的框只跳过距离损失，检测损失不受影响。

重新生成 6 列距离标签（教师模型直接推理，**不落 npy**）：

```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
python $P/tools/data/make_dist_labels.py --splits val --limit 20   # 试跑，写 /tmp 验证
python $P/tools/data/make_dist_labels.py                           # train+val 全量（默认 resume）
```

- 教师默认 `weights/yolo26x-depth.pt`（`--weights` 可换），imgsz 768；
- 输出 6 列 txt，前 5 列逐字节保留原标签；
- 统计写 `stats/detdist/{split}_summary.json` + `{split}_boxes.csv`，失败清单 `logs/failed_{split}.txt`。

---

## 4. 训练入口

### 4.1 detdist（`train_detdist.sh`）

```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace

# 改过数据集位置后：重写 data yaml（--smoke 顺带生成 20+10 张冒烟清单）
python $P/tools/data/make_data_yaml.py --smoke

# 看最终生效参数（不加载模型、不训练）
python $P/tools/train/train_detdist.py --dry-run

# 后台训练：PID → logs/detdist.pid，日志 → logs/detdist_<ts>.log，产物 → runs/detdist_pf
$P/train_detdist.sh start
$P/train_detdist.sh tail | status | stop

# 断点/继续训练：name 相同 + DETDIST_TAG 区分日志
$P/train_detdist.sh start --name detdist_pf --device 0
```

- 参数三层覆盖：`tools/train/train_detdist.py` 顶部 `DEFAULTS` < `configs/detdist_train.yaml` < 命令行（`--key value` 或 `key=value`）。
- 关键默认值：`imgsz 640`、`batch 96`、`epochs 400`、`device 0`、`lr0 0.001`、`dist_lambda 0.5`、`dist_huber_delta 0.1`、`close_mosaic 10`。
- 数据集只有一处定义：`configs/detdist_dataset.yaml` 的 `dataset_root`；train 读 `labels_dist`，val 也读 `labels_dist`（距离列不参与 mAP）。
- 不写 `.cache`：`libs/detdist/data.py` 自定义 `get_labels`，数据集目录保持干净。

### 4.2 detdepth（`train_coco_pf.sh`）

```bash
$P/train_coco_pf.sh start      # 后台，PID → logs/pf_train.pid
$P/train_coco_pf.sh status | tail | stop   # 不带子命令 = 前台直跑
```

- 数据：`configs/coco_pf_detdepth.yaml`；模型：`configs/detdepth_from_scratch.yaml`；
- 入口：`tools/train/train_detdepth_pf.py`（`trainer=DetDepthTrainer`，深度头从 `model_common/yolo26n-depth.pt` 迁移）；
- 产物：`runs/detdepth_coco_pf/`；日志 `logs/pf_train_<ts>.log`。

---

## 5. 结果在哪看

README 只记录目录与用法，不记录指标（避免和真实训练进度脱节）。每个 run 的结果都在自己的目录里：

- `runs/<name>/results.csv`：逐 epoch 的 `box/cls/dfl(/dist)`、`metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`；
- `runs/<name>/weights/{best,last}.pt`：权重；
- `runs/<name>/dist_eval.json`：detdist 的距离分桶报告（0–5 / 5–10 / 10–15 m 的 GT 中位、预测中位、偏差、MAE，以及整体 Spearman ρ）；
- `runs/<name>/*.png`：训练曲线与验证集拼图。

快速看所有 run 的最好一行：

```bash
cd hzh_workspace
python - <<'PY'
import csv, glob
for f in sorted(glob.glob("runs/*/results.csv")):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    b = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"] or 0))
    print(f"{f:44s} ep={b['epoch']:>4s} mAP50={b['metrics/mAP50(B)']} mAP50-95={b['metrics/mAP50-95(B)']}")
PY
```

或直接用 `tools/eval/eval_detdist.py --run runs/<name>` 一次性拿到 mAP + 距离分桶。

## 6. detdist 实现要点

- **不改仓库**：yaml 里仍写 `Detect`；`libs/detdist/model.py` 构造完 `DetectionModel` 后把 head 的类切成 `DetectDist`，并给每一级挂一个 1×1 conv（P3/P4/P5 各一个，共 +451 参数）。
- **输出**：每级 `dist` 与 box/cls 同 anchor 顺序；`sigmoid` 后乘 15 得米。训练时 `forward` 返回 `{"one2many": {boxes, scores, feats, dist}, "one2one": …}`。
- **损失**：`libs/detdist/loss.py` 在 `v8DetectionLoss` 的分配结果上，只对 `fg_mask & (d_norm >= 0)` 的 anchor 算 `Huber(sigmoid(pred), d_norm, δ=0.1)`；λ 由 `DETDIST_LAMBDA` 控制，dist 项不参与 one2many 的权重衰减。
- **数据**：`DistDataset` 覆盖 `get_labels` 读 6 列；`DistRandomPerspective`/`DistFormat` 在几何增强/格式转换时同步裁剪 `dist`，保证框与距离不串位。
- **评估**：`tools/eval/eval_detdist.py` 用 GT 框中心命中的 anchor 取预测距离中位数，按 0–5/5–10/10–15 分桶输出到 `runs/<name>/dist_eval.json`。

细节、改动清单、回退方法见 [`docs/detdist.md`](docs/detdist.md)。

---

## 7. 导出 / 量化 / 板端

| 目标 | 脚本 | 输出 |
|---|---|---|
| .pt → ONNX（整图） | `tools/export/pt_to_onnx.py`、`export_onnx_any.py`、`export_onnx_batch.py` | 2 输出 `output0`（检测+距离拼接） |
| .pt → ONNX（拆头） | `tools/split_pipeline*/export_onnx_split.py` | detdepth 10 输出 / detdist 12 输出 |
| ONNX → RKNN | `tools/convert/convert.py`（int8）、`official_convert.py`、`hybrid_multi2.py`（混合精度） | `*.rknn` |
| 一条龙 | `tools/split_pipeline/run_rknn_pipeline.sh`（detdepth）、`tools/split_pipeline_detdist/run_rknn_pipeline.sh`（detdist） | `models/<run>_<chip>/` + 预览图 |

- detdist 的拆头流水线把可调项集中在 `tools/split_pipeline_detdist/pipeline.env`，命令行可覆盖；`--print` 先看生效值。
- **int8 注意**：`dist_p3/p4/p5` 在 int8 模型里可能变成整型输出，板端要按 `zp/scale` 反量化；fp16 无此问题。
- 产物目录 `models/`：`split_pf_11_2`、`split_pf_11_2_3588`、`split_pf_v2`、`split_pf_v2_rk3576`、`split_pf_11_10_3588`（detdist 权重，12 输出）。

---

## 8. 评估与可视化

- `tools/eval/eval_detdist.py`：mAP（读 `results.csv`）+ 距离分桶；`--baseline` 可做两组 run 的对照。
- `tools/eval/eval_onnx.py`：onnxruntime 复现板端预测（字母框 + pad 114 + RGB/255），输出 `stem cls score x1 y1 x2 y2`。
- `tools/eval/dist_from_onnx.py` / `dist_both.py`：在 GT 框内取 ONNX 输出距离，产出 CSV/对照表。
- `tools/eval/acc_analysis.py`：int8 逐层量化误差（x86 模拟，fp32 vs quant 快照）。
- `tools/vis/vis_dist3.py`：一张图两色框（黄 GT / 红预测）+ 框上距离，顶部打印匹配框的距离 MAE。

---

## 9. 伪深度生成（detdepth 路线）

```bash
$P/gen_depth.sh --root /mnt/data2/COCO_PF --splits train val   # 例：补全 COCO_PF
```

- 教师默认 `model_common/yolo26n-depth.pt`，输出 `depth/<split>/<stem>.png`（uint16，毫米，`depth_scale=1000`）；
- 记录、格式契约、踩过的坑见 [`depth_gen/README.md`](depth_gen/README.md) 与 `depth_gen/coco_pf/report.md`。

---

## 10. 日志与环境

- 日志命名：`logs/{detdist,pf_train,split_*,dist_eval_*}…<ts>.log`；`*_latest.log` 是指向最新一条的软链；PID 文件为 `logs/{detdist,pf_train}.pid`。
- 必须的环境：
  ```bash
  PY=/home/kang/miniforge3/envs/yolo/bin/python
  export PYTHONPATH=<repo>:<repo>/hzh_workspace/libs
  export YOLO_CONFIG_DIR=<repo>/hzh_workspace/env/cfg
  ```
- 为什么：site-packages 的 ultralytics 太旧（8.4.142，无 Depth 头）；仓库版是 8.4.192。`YOLO_CONFIG_DIR` 防止往 home 写配置/字体缓存。
- `env/pylibs/` 是额外的私有依赖目录（numpy 等），需要时由 `PYTHONPATH` 优先加载。

---

## 11. 维护约定

1. **`tools/` 只放可复用脚本**；一次性实验脚本跑完就删（删前先 `grep -rn <名字>` 确认没有引用）。
2. **`runs/`、`models/`、`env/` 不要动**；删任何产物前先确认没有板端/文档引用。
3. **不往数据集里写东西**：detdist 不写 `.cache`；跑 ultralytics 原生 `YOLODataset`（如 detdepth、旧 run 的 val）会在 `labels/` 下生成 `*.cache`，属于工具行为，事后可删。
4. **新链路命名**：入口 `<name>.sh`、实现 `libs/<name>/`、配置 `configs/<name>_*.yaml`、日志 `logs/<name>_*.log`、文档 `docs/<name>.md`。
5. **文档就地更新**：改参数默认值 → 改 `tools/train/*.py` 的 `DEFAULTS`；改目录结构 → 更新本 README 第 1 节与对应 `docs/`。
6. 历史目录/血缘的详细分析见 `WORKSPACE_ANALYSIS.md`（若存在）；旧的 `archive/`、`calib/`、`reports/`、`plan_v2/` 已清理，结论保留在文档里。
