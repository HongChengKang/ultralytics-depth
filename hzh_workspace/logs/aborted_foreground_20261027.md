> 2026-10-28：`runs/_aborted_foreground/` 的 67 MB 数据已删除；本文件是原目录 README 的备份。

# `_aborted_foreground`：2026-10-27 的启动/调参失败记录

**这不是正式训练结果，不要拿这里的权重做部署或对比。**

2026-10-27 13:42–13:54，COCO_PF 正式训练（v1，13:57 启动）之前，在同一台 A6000 上做了三次启动尝试；
前两次双卡 OOM，第三次单卡跑通 2 epoch 后被人为停掉，三个目录被移到 `runs/_aborted_foreground/`，
以便和正式 run 区分。合计约 67 MB。

> 下表中的 `pf_detdepth_from_scratch*` 是 10-27 当时的目录名，均已随本目录一起删除；
> 正式 v1 的当前路径是 `runs/detdepth_coco_pf_old`，当前 `runs/detdepth_coco_pf` 是 v2。

| 目录 | batch | device | 起止 | 结果 | 权重 |
|---|---:|---|---|---|---|
| `pf_detdepth_from_scratch/` | 256 | 0,1 | 13:42 | 首批/建图阶段失败，无 `results.csv` | `weights/` 空 |
| `pf_detdepth_from_scratch-2/` | 128 | 0,1 | 13:46 | 首批失败，无 `results.csv` | `weights/` 空 |
| `pf_detdepth_from_scratch-3/` | 80 | 0 | 13:51–13:52 | 跑完 2 epoch，mAP50≈9e-05，随后被停 | `best.pt` / `last.pt` 各 33,069,040 B |

第三次的 `args.yaml` 里 `epochs=400`，但只完成 2 epoch；`results.csv` 只有 2 行。
正式 v1 是 13:57 启动的 `runs/detdepth_coco_pf_old`（训练时原名 `pf_detdepth_from_scratch`；136 epoch，
best mAP50 0.75214 / mAP50-95 0.48695），两者不是同一次 run。

## 处理建议

- 保留价值：只用于说明「双卡 batch 256/128 OOM → 单卡 batch 80 可行」的调参过程；
- 没有部署/复现价值，也没有完整训练曲线；
- 确认不需要这段过程记录后，可整体删除（约 67 MB）。删之前先确认没有别的脚本引用它。
