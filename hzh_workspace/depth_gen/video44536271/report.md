# COCO_PF 新增 1000 张（video44/53/62/71）深度图生成报告

> 归档说明（2026-10-28）：本目录已并入 `depth_gen/video44536271/`；统一入口为根目录 `gen_depth.sh`。正文保留当时的生成记录与数字，路径以根 README 为准。


生成时间：2026-10-27 18:42–18:44 · 机器：train-server（GPU 0）

## 1. 任务与约束
- 为 `/mnt/data2/COCO_PF` 中**新增**的四个视频序列 `video44 / video53 / video62 / video71` 生成逐像素深度图。
- 只补缺失、**不覆盖**已有；格式与已有 depth 完全一致；产物直接写入数据集 `depth/{train,val}`。
- 不改动 `images/`、`labels/`、已有 `depth/*.png`；日志/报告放 `hzh_workspace/depth_gen/video44536271/`。

## 2. 数据现状（生成前）
| split | images | depth | 缺失 | 缺失构成 |
|---|---|---|---|---|
| train | 3188 | 2388 | 800 | video44/53/62/71 各 200 |
| val | 672 | 472 | 200 | video44/53/62/71 各 50 |

新增图片均为 1920×1080 PNG（1.4–2.8 MB/张），旧数据（camera1-1/2-1/3-1/4-1、output_fpv）完整。

## 3. 格式契约（与已有 depth 一致，已核）
- 16-bit 无损 PNG（写 uint16，读回 PIL 给 int32，字节一致），非 8-bit 可视化图。
- 单位**毫米**（`depth_scale=1000`），与 RGB **同分辨率**、同名 `<stem>.png`，1:1 一一对应。
- 有效像素（>0）占比 1.0000。

## 4. 脚本与命令
复用 `hzh_workspace/tools/data/gen_pseudo_depth.py`（**未改一行**；内部 `if not (out/f"{stem}.png").exists()` 天然跳过已有文件）。
教师模型：`/home/kang/workspace/projects/yolo-depth/ultralytics-depth/model_common/yolo26n-depth.pt`
（12,998,295 B，md5 `4a4d810a0a783f68f9a3760f36c117be`）

```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
R=/home/kang/workspace/projects/yolo-depth/ultralytics-depth
cd $P
export PYTHONPATH=$P/env/pylibs:$R YOLO_CONFIG_DIR=$P/env/cfg \
       MPLCONFIGDIR=$P/env/cfg/mpl TORCH_HOME=$P/env/cfg/torch
python3 tools/data/gen_pseudo_depth.py \
  --root /mnt/data2/COCO_PF --splits train val \
  --imgsz 640 --batch 16 --device 0 --weights $R/model_common/yolo26n-depth.pt
```

**踩坑**：脚本被移动到 `tools/data/` 后，内部 `REPO` 由 `__file__.parents[2]` 推出，默认 `--weights` 变成 `hzh_workspace/model_common/yolo26n-depth.pt`（不存在），ultralytics 会转而从 GitHub 下载并**卡死**（进程 `do_sys_poll`、GPU 0%，`model_common/.yolo26n-depth.pt.*.part` 就是证据）。**必须显式传 `--weights`**。

## 5. 结果
| 项 | 值 |
|---|---|
| 生成数量 | **1000**（train 800 + val 200） |
| 失败 | **0**（日志无 error/traceback） |
| 耗时 | train 57.8 s + val 13.5 s ≈ **71 s**（约 **0.07 s/张**，不含加载模型） |
| 体积 | **+1.4 GB**（单张约 1.4 MB） |
| 生成后 | train 3188/3188、val 672/672，**缺失 0、多余 0** |

## 6. 验证
1. **零覆盖**：生成后 `find depth -mmin -20` 恰好 **1000** 个文件，旧文件 mtime 未变。
2. **口径一致性**：随机取 3 张已有 depth 的图（camera1-1_frame_000000..2）用同一脚本重跑到 /tmp 逐像素比对——
   max|Δ| 17–18 mm、mean 1.4–3.4 mm、中位相对误差 4e-5~2e-4（纯浮点抖动）。
3. **统计合理性**（单位 mm）：
   | 序列 | n | 最近距离（中位） | 最远距离（中位） | 有效像素 |
   |---|---|---|---|---|
   | video44 | 250 | 2848 | 29910 | 1.0000 |
   | video53 | 250 | 479 | 22571 | 1.0000 |
   | video62 | 250 | 904 | 27328 | 1.0000 |
   | video71 | 250 | 1260 | 28086 | 1.0000 |
   | 旧 camera1-1 | 239 | 1869 | 32617 | 1.0000 |
   video53 的 479 mm 来自贴着镜头的机身/支架，属正常。

## 7. 产物
- `depth_gen/video44536271/report.md`（本文件）
- `depth_gen/video44536271/compare.jpg`（4 个序列 × train/val 抽样，RGB | 深度伪彩 对照）
- `depth_gen/video44536271/logs/gen.log`
- 数据集：`/mnt/data2/COCO_PF/depth/{train,val}/video{44,53,62,71}_*.png` 共 1000 张

脚本的"跳过已存在"是幂等的：以后再加图片，直接再跑同一条命令即可，只会补新增。
