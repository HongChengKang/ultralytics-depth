# COCO_PF 新增图片深度图生成报告

> 归档说明（2026-10-28）：本目录已并入 `depth_gen/coco_pf/`；生成脚本现为 `tools/data/gen_pseudo_depth.py`，统一入口为根目录 `gen_depth.sh`。正文保留当时的生成记录与数字，路径以根 README 为准。


日期：2026-10-27　　机器：train-server（RTX A6000，device 0）
代码根：`/home/kang/workspace/projects/yolo-depth/ultralytics-depth`
数据集：`/mnt/data2/COCO_PF`（只动 depth/，未动 images/、labels/，未新建任何目录）

---

## 一、侦察结论

### 1. 目录结构
```
/mnt/data2/COCO_PF/{images,labels,depth}/{train,val}
```

### 2. 已有 depth 的格式契约（从现有文件实测）

| 项 | 值 |
|---|---|
| 扩展名 | `.png`（16-bit 灰度，PIL 读成 mode `I`/int32） |
| dtype/位深 | **uint16**（0–65535） |
| 单位 | **毫米**（`depth_scale: 1000`，1000 = 1 m） |
| shape | **与对应 RGB 完全同分辨率**（如 camera2 为 2560×1440、output_fpv 为 1920×1080） |
| 命名 | 与 RGB **同名**，仅换后缀：`<stem>.png` |
| 有效像素 | 现有文件全部 `>0` 占比 100%（脚本 clip 到 ≥1，避免 0 被判为无效） |

### 3. 教师模型
```
/home/kang/workspace/projects/yolo-depth/ultralytics-depth/model_common/yolo26n-depth.pt
12,998,295 B   md5 4a4d810a0a783f68f9a3760f36c117be
```

### 4. 生成脚本
**复用现成脚本，未新写、未改一行**：
`/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace/tools/data/gen_pseudo_depth.py`
（其 `--root` 默认值本来就是 COCO_PF，`--weights` 默认就是上面的教师模型）

脚本逻辑：`YOLO(weights).predict(..., imgsz=640)` → `r.depth.data`（米，原图分辨率）
→ `clip(d*1000, 1, 65535).astype(uint16)` → `depth/<split>/<stem>.png`；
**生成前先过滤 `if not (out_dir/f{stem}.png).exists()`，天然不覆盖**。

### 5. 待生成数量（生成前统计）

| split | images | 已有 depth | **待生成** | 多余 depth |
|---|---|---|---|---|
| train | 2388 | 2088 | **300** | 0 |
| val | 472 | 397 | **75** | 0 |
| 合计 | 2860 | 2485 | **375** | 0 |

缺失的全部是**新加的两个相机序列**：train 的 `camera2-1`(142) + `camera4-1`(158)，
val 的 `camera2-1`(35) + `camera4-1`(40)。原有 camera1/camera3/output_fpv/COCO 编号图均已完整。

### 6. 可写性
`/mnt/data2/COCO_PF/depth` **可写**（792 G 可用）→ 直接写入数据集原目录，**无需手动合并**。

---

## 二、生成过程

### smoke test（5 张，先跑后验）
从待生成列表取 `camera2-1_frame_000008…000012`，输出到 `/tmp/depth_smoke`，与已有文件对照：

| 项 | 新生成 | 已有（camera1-1） | 一致 |
|---|---|---|---|
| 格式 | PNG / mode I / 16-bit | PNG / mode I / 16-bit | ✅ |
| shape | 1440×2560（= RGB） | 1440×2560（= RGB） | ✅ |
| 数值范围 | 1093–33971 | 1717–31712 | ✅ 同量级（mm） |
| 有效像素 | 1.0000 | 1.0000 | ✅ |
| 命名 | `<stem>.png` | `<stem>.png` | ✅ |

### 全量
```bash
P=/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
R=/home/kang/workspace/projects/yolo-depth/ultralytics-depth
cd $P
export PYTHONPATH=$P/env/pylibs:$R/hck_workspace/detdepth:$R YOLO_CONFIG_DIR=$P/env/cfg \
       MPLCONFIGDIR=$P/env/cfg/mpl TORCH_HOME=$P/env/cfg/torch
python3 tools/data/gen_pseudo_depth.py --root /mnt/data2/COCO_PF --splits train val --imgsz 640 --batch 32 --device 0
```

| 项 | 值 |
|---|---|
| 实际生成 | **375**（train 300 + val 75） |
| 失败 | **0**（进程 EXIT=0，无异常；`failed.txt` 为空） |
| 总耗时 | **44 s**（train 33.8 s + val 8.3 s + 模型加载约 2 s） |
| 单张平均 | **≈ 0.11 s** |
| 新增体积 | 0.9 GB（单张约 2.4 MB，2560×1440 无损 PNG） |

只覆盖缺失项的证据：生成后 `find depth -mmin -30` 恰为 **375** 个文件，其余 2485 个旧文件 mtime 未变。

---

## 三、验证

### 1. 数量对齐（生成后）
| split | images | depth | 缺失 | 多余 |
|---|---|---|---|---|
| train | 2388 | **2388** | **0** | **0** |
| val | 472 | **472** | **0** | **0** |

一一对应，无多余、无缺失。

### 2. 统计对比（全量 depth 文件）

| 组 | 文件数 | min 中位数 | min p1 | max 中位数 | max p99 | 有效像素(>0) |
|---|---|---|---|---|---|---|
| 新生成 train（camera2/4） | 300 | 519.5 | 387.0 | 28,943.0 | 36,392.7 | 1.0000 |
| 已有 train（camera1-1） | 191 | 1,896.0 | 1,561.0 | 32,656.0 | 52,576.5 | 1.0000 |
| 新生成 val（camera2/4） | 75 | 501.0 | 375.4 | 28,407.0 | 34,188.1 | 1.0000 |

单位、量级一致（几百 mm 到数十米）；min 偏小是因为 camera2/4 视角里紧贴镜头的机身/支架，属真实近景。

### 3. 口径一致性（关键）
把 **5 张已有 depth 的图**（`camera1-1_frame_000000…000004`）用同一脚本、同一教师模型重跑到临时目录，与数据集里**原有 depth 逐像素比对**：

| 图 | max\|diff\| (mm) | mean\|diff\| (mm) | 相对误差中位数 | 差异像素占比 |
|---|---|---|---|---|
| 000000 | 18 | 2.40 | 0.0001 | 68% |
| 000001 | 17 | 1.44 | 0.0001 | 61% |
| 000002 | 11 | 0.97 | 0.0000 | 51% |
| 000003 | 16 | 1.51 | 0.0001 | 63% |
| 000004 | 19 | 3.71 | 0.0002 | 78% |

**最大差 19 mm（1.9 cm），平均差 1–4 mm，相对误差 1e-4 量级** —— 差异只来自 GPU 推理/插值的浮点抖动，
说明新生成与已有 depth **完全同口径**，可以混用。

### 4. 可视化对照
`compare.jpg`（10 组：左 RGB，右深度伪彩 turbo，2560×1890）
路径：`hzh_workspace/depth_gen/coco_pf/compare.jpg`

---

## 四、结论

- ✅ 格式契约完全一致（uint16 PNG、毫米、原图分辨率、同名 `.png`），可直接与旧 depth 混用训练。
- ✅ 只补缺失：375 张全部为新增，2485 张旧文件零改动、零覆盖。
- ✅ 与已有 depth 逐像素一致（≤19 mm 抖动），口径无疑问。
- ✅ 产物已直接写入 `/mnt/data2/COCO_PF/depth/{train,val}`，**不需要任何手动合并**。
- ⚠️ 数据集新增 0.9 GB；若后续还要批量补，建议先看 `/mnt/data2` 余量（当前 792 G）。

## 五、复现/续跑

增量脚本天然幂等：再跑一次只会处理之后新加的图片。

```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace
./gen_depth.sh --root /mnt/data2/COCO_PF --splits train val --imgsz 640 --batch 32 --device 0
```
