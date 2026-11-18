# metdist — 检测 + 米制距离回归（正式流程）

原 `detdist` / `detdist_v2` 系列实验收编后的正式名。自包含包 `libs/metdist`
（合并 v1 数据/损失基类 + v2 深距离塔），旧 checkpoint 的 `detdist_v2.*` pickle
路径通过包内别名自动兼容。`detdepth`（稠密深度路线）独立保留，互不影响。

## 模型与输出
- YOLO26n + 每级 Conv3x3(ch,32)+Conv3x3(32,1) 距离塔（+130k 参数，总 2.63M）
- 头输出 log 深度；导出图内 `exp(clamp(logit, log0.2, log15))` → 12 输出，
  dist_p3/p4/p5 直接是米；板端聚合：框内 anchor 中位数，<5 m 取 20 分位
- 推理预算：3588 int8 NPU ~40.7 ms / 端到端 ~43.6 ms（实测）

## 当前最优权重与配方
- 权重: `runs/detdist_pf_v2_asym3/weights/last.pt`（历史 run 名保留）
- 配方: `configs/metdist_best.yaml`（tau 0.25 / near_weight 3 / lr 0.002 /
  patience 0 / save_period 10）
- 验收: own_video val 近场 MAE 0.28 m、<=0.5 m 比例 82%、rho 0.95；
  新機位泛化仍是短板（5 张留出图 MAE ~1.2 m），部署近场建议几何兜底
- 标签: `labels_dist_own`（own 域米制监督）；`labels_dist_cal`（全量校正，
  供序损失/评估）；生成链 make_dist_labels → calibrate_dist_labels →
  make_own_dist_labels

## 命令
    ./train_metdist.sh start --config configs/metdist_best.yaml   # 训练
    ./train_metdist.sh tail|status|stop
    tools/train/train_metdist.py --dry-run                        # 只看参数
    tools/eval/eval_metdist.py --logdepth --near-pct 20           # own val 评估
    tools/split_pipeline_metdist/run_rknn_pipeline.sh             # 3576
    tools/split_pipeline_metdist/run_rknn_pipeline.sh --platform rk3588 \
        --out models/split_pf_11_17_3588                          # 3588
    tools/data/make_data_yaml.py [--smoke]                        # 数据 yaml

## 命名映射（旧 → 新）
    libs/detdist + libs/detdist_v2 → libs/metdist
    train_detdist*.sh / tools/train/train_detdist*.py → train_metdist.sh /
        tools/train/train_metdist.py
    configs/detdist_* → configs/metdist_*（最优配方 metdist_best.yaml）
    tools/eval/eval_detdist.py / val_ownvideo_infer.py → tools/eval/eval_metdist.py
    tools/split_pipeline_detdist → tools/split_pipeline_metdist
    docs/detdist.md → docs/metdist.md

## 已删除（2026-11-18 清理）
libs/detdist、libs/dist_plan、libs/detdist_v2、全部 configs/detdist*、
旧训练入口/启动器、旧 eval 脚本（eval_detdist/eval_onnx/dist_both/
dist_from_onnx/demo_distance_cd/detdist_subset_report/val_ownvideo_infer）、
make_gen_data.py、split_pipeline_detdist、datasets/detdist_{gen,smoke}、
旧模型产物 models/split_pf_{11_10_3588,11_11_3576,11_2,11_2_3588,v2,v2_rk3576}；
da3（Depth Anything 3 源码/venv/6.7G 权重, 共 6.9G, 2026-11-18 应要求删除）。
保留: runs/*（含权重）、logs/*、models/split_pf_11_17_*、detdepth 全链路、
depth_gen、tools/eval/acc_analysis.py。
