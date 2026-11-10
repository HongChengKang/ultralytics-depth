# depth_gen：伪深度生成记录与入口

本目录集中保存「用深度 teacher 给检测数据集补逐像素伪深度」的历史记录，并提供根目录
[`gen_depth.sh`](../gen_depth.sh) 作为可重复使用的入口。

数据集本身写在 `/mnt/data2/.../depth/{train,val}/`，**不在本目录**；这里只放报告、日志、对照图和校验脚本。

## 1. 目录与两次生成

```
depth_gen/
├── README.md                       # 本文件
├── coco_pf/                        # COCO_PF 的两次增量生成
│   ├── report.md                   #   2026-10-27 13:03 增量 375 张（服务 v1）
│   ├── compare.jpg                 #   RGB / 深度伪彩对照
│   ├── failed.txt                  #   空 = 0 失败
│   ├── verify.py                   #   数量/格式/重跑一致性校验
│   └── logs/
│       ├── initial_full_gen.log    #   首次全量 2088 train + 397 val（从旧 archive 移入）
│       ├── gen.log                 #   增量 300 train + 75 val，EXIT=0
│       └── run_start_unix.txt / run_end_unix.txt
└── video44536271/                  # COCO_PF 新增四个视频序列
    ├── report.md                   #   2026-10-27 18:42 增量 1000 张（服务 v2）
    ├── compare.jpg
    └── logs/gen.log                #   800 train + 200 val，EXIT=0
```

| 批次 | 时间 | 服务对象 | 新增 depth | 生成后 train/val | 耗时 | 体积 |
|---|---|---|---|---:|---:|---:|---:|
| 首次全量 | 2026-10-27 上午 | COCO_PF 基线 | 2088 + 397 | 2088 / 397 | — | — |
| `coco_pf` 增量 | 2026-10-27 13:03–13:05 | `runs/detdepth_coco_pf_old`（v1） | 300 + 75 | 2388 / 472 | 44 s | +0.9 GB |
| `video44536271` 增量 | 2026-10-27 18:42–18:44 | `runs/detdepth_coco_pf`（v2） | 800 + 200 | 3188 / 672 | 71 s | +1.4 GB |

两次增量都是 **0 失败**，且只写缺失文件，旧 depth 的 mtime 未变。

> run 目录已于 2026-10-28 按数据集统一改名：当前 v1 为 `detdepth_coco_pf_old`、v2 为 `detdepth_coco_pf`；
> 生成当时 v1 叫 `pf_detdepth_from_scratch`、v2 叫 `pf_detdepth_from_scratch_v2`。

> 报告正文写于目录合并之前，里面的 `depth_gen_coco_pf/`、`depth_gen_video44536271/`、
> `tools/gen_pseudo_depth.py` 等是当时的路径；当前路径以本 README 和根 README 为准。
> 报告正文保留原始记录，未改动结论与数字。

## 2. 格式契约（下游训练依赖）

- 16-bit 无损 PNG（`uint16`），不是 8-bit 可视化图；
- 单位 **毫米**，`depth_scale: 1000`；
- 与 RGB **同分辨率、同 stem**：`<root>/depth/<split>/<stem>.png`；
- 有效像素写 `>0`（脚本 clip 到 `[1, 65535]`），避免 0 被当成无效深度。

## 3. 复现 / 给新数据补深度

```bash
cd /home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace

# 最常用：给 COCO_PF 的 train/val 补新增图片
./gen_depth.sh --root /mnt/data2/COCO_PF --splits train val

# 只补 val，换 batch/device
./gen_depth.sh --root /mnt/data2/COCO_PF --splits val --batch 16 --device 0

# 换 teacher（默认是仓库根 model_common/yolo26n-depth.pt）
./gen_depth.sh --root /mnt/data2/COCO_PF --splits train \
  --weights /path/to/other_teacher.pt
```

脚本是幂等的：已存在的 `<stem>.png` 会跳过，重复跑只补新增图片。
底层实现见 [`tools/data/gen_pseudo_depth.py`](../tools/data/gen_pseudo_depth.py)。

## 4. 关键坑（已踩过）

`tools/data/gen_pseudo_depth.py` 内部用 `Path(__file__).parents[3]` 推仓库根，当前版本是对的；
但如果脚本再被移动，默认 `--weights` 可能指向不存在的 `hzh_workspace/model_common/yolo26n-depth.pt`，
ultralytics 会转去 GitHub 下载并卡死（旧 `model_common/*.part` 就是证据）。
因此 [`gen_depth.sh`](../gen_depth.sh) 会显式补上仓库根的 teacher 路径，不依赖脚本内部默认值。

## 5. 与旧 archive 的关系

旧 `archive/logs_old/gen_pseudo_depth_coco_pf.log` 是首次全量生成的日志，已复制为
`coco_pf/logs/initial_full_gen.log`；`archive/` 其余内容（VisDrone 单头混合精度/量化中间件、
1.9 GB 逐层快照、旧工具）与三次检测训练无关，已于 2026-10-28 删除。
