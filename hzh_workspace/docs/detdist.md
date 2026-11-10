# detdist：检测头多尺度距离回归（框级距离标签 → 训练 → 评估）

在工作区 `hzh_workspace/` 内、**不修改仓库 `ultralytics/` 源码**的前提下，给 YOLO26n
检测头加「每级一个 1×1 conv 的距离通道」，用 `labels_dist/` 的 6 列标签做框级距离监督：

- 距离通道输出 `sigmoid` → `d_norm ∈ [0,1]`，`d_m = d_norm × 15`
- `d_norm = -1` 的框是距离 ignore：**检测损失照常、距离损失不参与**（不转 0/1）
- 损失 `L = L_box + L_cls + L_dfl + λ·L_dist`；`L_dist` 只对「正样本且 `d_norm ≥ 0`」的
  anchor 计 Huber(`sigmoid(pred)`, `d_norm`, δ=0.1)，δ=0.1 ≈ 1.5 m；λ 默认 0.5
- 从 0 训练，不加载任何检测权重

工作区总览见 [`../README.md`](../README.md)。

---

## 1. 落位（按工作区功能分区）

| 功能 | 路径 |
| --- | --- |
| 启动器（与 `train_coco_pf.sh` 同级） | `hzh_workspace/train_detdist.sh` |
| 数据集定义（**唯一改这里**） | `hzh_workspace/configs/detdist_dataset.yaml`（`dataset_root` 一行） |
| 生成 data yaml | `hzh_workspace/tools/data/make_data_yaml.py` → `configs/detdist_data.yaml` |
| 模型 / 训练 yaml | `hzh_workspace/configs/detdist_{model,train}.yaml` |
| 可复用实现（注入用） | `hzh_workspace/libs/detdist/{head,data,loss,model,trainer}.py` |
| 旧 checkpoint 兼容层 | `hzh_workspace/libs/dist_plan/{head,model}.py`（只有 re-export，见 §6） |
| 一步式距离标注 | `hzh_workspace/tools/data/make_dist_labels.py` |
| 训练入口 | `hzh_workspace/tools/train/train_detdist.py` |
| 评估脚本 | `hzh_workspace/tools/eval/eval_detdist.py` |
| RKNN 部署流水线 | `hzh_workspace/tools/split_pipeline_detdist/` |
| 训练输出 | `hzh_workspace/runs/<name>/`（`--project` 默认 `runs`） |
| 日志/PID | `hzh_workspace/logs/detdist_*.log`、`logs/detdist.pid` |
| 标注统计 | `hzh_workspace/stats/detdist/{split}_{summary.json,boxes.csv}` |
| 本文档 | `hzh_workspace/docs/detdist.md` |

仓库 `ultralytics/` 一行未改：yaml 里仍写 `Detect`，`DistModel` 构造完把 `head.__class__`
换成 `DetectDist` 再加距离分支（注入式，删掉本链路即完全回退）。

## 2. 数据集：只有一个事实来源

`configs/detdist_dataset.yaml` 里只有一行真正重要的东西：

```yaml
dataset_root: /mnt/data2/hzh/PF_11.9
```

改了它，再跑一次 `tools/data/make_data_yaml.py`，所有路径跟着变。**不再有数据集镜像**：

- 图像：`<dataset_root>/images/{train,val}`，直读；
- 标签：`<dataset_root>/labels_dist/{train,val}`（6 列），由 `DistDataset(labels_root=…)` 解析，
  不依赖 `images/ → labels/` 的目录替换；
- 验证集也走 `DistDataset`，所以**不需要**再复制一份 5 列标签；
- `DistDataset.get_labels` 是自定义实现，**不会往数据集里写 `.cache`**（ultralytics 原版会）。

### 2.1 生成 6 列距离标签（一步，不落 npy）

```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
python tools/data/make_dist_labels.py --splits val --limit 20 \
    --output-dir /tmp/labels_dist_smoke     # 冒烟，20 张
python tools/data/make_dist_labels.py       # train+val 全量，默认 resume
```

- 教师默认 `weights/yolo26x-depth.pt`（imgsz 768），`--weights` 可换；
- 逐图推理 → 框内有效像素（`0 < d ≤ max_dist`）中位数 → 直接写
  `cls cx cy w h d_norm`，**不写 `.npy`、不写深度图、不占额外磁盘**；
- 前 5 列逐字节保留；`d_norm = -1` 的框照样写出（检测标签不丢）；
- 参数：`--max-dist 15`、`--min-valid-pixels 10`、`--overwrite`、`--limit`、`--batch`、`--device`；
- 统计：`stats/detdist/{split}_summary.json`、`{split}_boxes.csv`；失败清单 `logs/failed_{split}.txt`；
- 每张图会打印一行 sanity check（`min/med/max`、`>0` 比例、`>max_dist` 比例），中位数不在
  0.1–200 之间会警告「教师输出可能不是米制」。

## 3. 命令

```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace

# 0) 生成 data yaml（改过 dataset_root 后必跑）；--smoke 顺带生成 20+10 张冒烟清单
python tools/data/make_data_yaml.py --smoke

# 1) 看最终生效参数（不加载模型、不训练）
python tools/train/train_detdist.py --dry-run

# 2) 冒烟：20 train + 10 val，1–2 epoch @320
python tools/train/train_detdist.py --data configs/detdist_smoke_data.yaml \
    --epochs 2 --imgsz 320 --batch 8 --workers 2 --name detdist_smoke --close_mosaic 0

# 3) 正式训练（后台，PID 写 logs/detdist.pid）
./train_detdist.sh start                      # 默认 400 epoch, imgsz 640, batch 96
./train_detdist.sh start --epochs 8           # 任意 key=value 覆盖
DETDIST_TAG=baseline ./train_detdist.sh start \
    --name detdist_baseline --dist_lambda 0 --device 1   # 同期 baseline（独立 pid/log）

# 4) 观察 / 停止
./train_detdist.sh status
./train_detdist.sh tail
./train_detdist.sh stop

# 5) 评估：mAP（results.csv）+ 距离分桶
python tools/eval/eval_detdist.py --run runs/detdist_pf --baseline runs/detdist_baseline
```

## 4. 评估输出（`runs/<name>/dist_eval.json`）

- `map.best / map.last`：从 `results.csv` 取 mAP50、mAP50-95（best 按 mAP50-95 最大）
- `distance.overall`：`n`、`spearman`、`mae_m`、`median_abs_err_m`、`signed_bias_m`
- `distance.buckets`：按**标签** `d_m` 分 0-5 / 5-10 / 10-15 m，各桶
  `n / gt_median_m / pred_median_m / signed_bias_m / mae_m`
- `--baseline` 时多一个 `delta_vs_baseline`（mAP50、mAP50-95 差值）

注意：衡量的是「相对教师伪距离标签的一致性」，不是米制精度真值。

## 5. 参数在哪改（三层覆盖）

覆盖顺序：**脚本 `DEFAULTS` < `configs/detdist_train.yaml` < 命令行**。

- 唯一默认值表：`tools/train/train_detdist.py` 顶部 `DEFAULTS`，含 `imgsz`（顺序 **H, W**）、
  `project: "runs"`、`name: "detdist_pf"`、`epochs/batch/device/workers/optimizer/lr0/lrf/
  warmup_epochs/patience/seed/deterministic/cache/cos_lr/close_mosaic/translate/scale/
  fliplr/flipud/hsv_h/hsv_s/hsv_v/erasing/dist_lambda/dist_huber_delta/exist_ok/plots`，
  以及 `model/data/labels_root` 三个路径（`labels_root` 留空 = `<dataset_root>/labels_dist`）。
- `configs/detdist_train.yaml` 只写覆盖项，当前为空（只剩注释示例）。
- CLI：`--name detdist_pf_2 --epochs 100 --imgsz "[640, 384]" --batch 64`，或 `key=value`；
  未识别的键透传给 ultralytics（`resume`、`save_period` 等）；`--config <file>` 整体换 yaml。
- `--dry-run` 打印 `effective training settings` 表，逐项标来源（`DEFAULTS` / yaml / `CLI`），
  并打印 `model/data/labels_root` 绝对路径与 `run dir`。

## 6. 运维注意

- 加载 `runs/<name>/weights/*.pt` 的脚本必须能 import 本包：
  `PYTHONPATH=<workspace>/libs:<repo>`（启动器已导出）。
- 2026-11-10 之前训出的 checkpoint 在 pickle 里引用旧模块名 `dist_plan.head`；
  `libs/dist_plan/` 是**只为这类旧权重保留的 re-export 兼容层**，新代码一律 `from detdist…`。
- 距离分支开销：每级 +1 通道，共 **+451** 参数。
- 吞吐（A6000，480×320，batch 96）：约 28 s/epoch（43 iter），400 epoch ≈ 3.1 h；640 输入约为其 2.3×。
- 历史参考（2026-11-10，8 epoch @480×320）：mAP50 0.4431 vs baseline 0.3918；
  距离 `rho=0.505`、MAE 2.411 m。
- `box_dist_labels/`（npy 两段式标注）已由 `tools/data/make_dist_labels.py` 取代并删除；
  `labels_dist/` 本次未重算，内容保持一致。
