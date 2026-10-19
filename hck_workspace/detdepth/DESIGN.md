# YOLO26 检测 + 深度 联合模型（detdepth）设计

日期：2026-10-16

## 目标

一个共享 backbone + neck 的 YOLO26 模型，一次前向同时输出：

- 目标检测框（单类 person；本地 VisDrone 标注全部为类别 0）
- 稠密深度图，以及每个框的框内深度（框内像素深度中位数）

## 约束与已验证的前提

- 只有检测标注（`/mnt/data2/VisDrone数据集`，images/labels 的 train/val/test）。深度监督来自教师模型
  `model_common/yolo26n-depth.pt` 生成的伪标签。
- 已在 200 张 val 图上验证教师：框纵向位置与框内深度的 Spearman 中位数 -0.92（画面越靠下越近，95% 图符合，0% 反向），
  说明远近排序可靠；但深度范围只有 ~3–20 m，米制尺度不可信。**因此本模型的深度定义为相对深度**，训练用尺度不变损失。
- 所有新代码放在 `hck_workspace/detdepth/`，只通过子类化 ultralytics 现有类实现，不修改 `ultralytics/` 包。
- 先只支持 PyTorch 推理；导出（ONNX/TensorRT）不在本期范围，但头部输出结构不做阻碍导出的设计。

## 文件结构

```
hck_workspace/detdepth/
├── DESIGN.md              本文档
├── gen_pseudo_depth.py    步骤 1：教师模型生成伪深度标签
├── visdrone-detdepth.yaml 数据集配置
├── detdepth.py            模型头 / 模型 / 损失 / 数据集 / Trainer / Validator / Predictor
├── train.py               训练入口
└── predict.py             推理入口（输出框 + 框内深度）
```

`detdepth.py` 是单一模块，所有类彼此紧耦合、且都是对现有类的薄子类，拆成多个文件只会增加 import 噪音。

## 1. 伪深度标签生成（`gen_pseudo_depth.py`）

- 用 `yolo26n-depth.pt`（imgsz=640）推理 `images/{train,val}` 全部图片，`Results.depth` 已是原图分辨率（米）。
- 保存为 16 位 PNG（毫米，`depth_scale: 1000`）到 `/mnt/data2/VisDrone数据集/depth/{train,val}/<同名>.png`，
  与现有 `DepthDataset._depth_path_for` 的 images→depth 目录约定一致，可直接复用 `load_depth`。
- 已存在的文件跳过，可中断重跑。

## 2. 数据集配置（`visdrone-detdepth.yaml`）

`path: /mnt/data2/VisDrone数据集`，`train/val/test` 同官方 `VisDrone.yaml`，单类 `0: person`，`depth_scale: 1000`。

## 3. 模型

- **结构**：用官方 `yolo26.yaml`（`nc=10`）构建 `DetectionModel`，backbone+neck 第 0–22 层不变。
- **组合头 `DetectDepth(Detect)`**：继承 `Detect`，额外持有一个子模块 `dhead = Depth(c_mid=256, ch)`，
  输入同样是第 16/19/22 层（P3/P4/P5）。
  - 训练：返回 `Detect` 的预测 dict，并增加键 `"depth"`（(B,1,H/4,W/4)）。
  - 推理：返回 `(y, preds)`，其中 `preds["depth"]` 为深度图；`y` 与原 `Detect` 完全一致，检测后处理零改动。
- **构建方式**：`DetDepthModel(DetectionModel)` 先按 `yolo26.yaml` 正常构建（stride、bias_init 都走原逻辑），
  再把 `model[-1]` 替换为同参数的 `DetectDepth` 并拷贝原 head 的 state_dict 与 `f/i/type/stride` 属性。
  这样不需要改 `parse_model`。
- **初始化**：
  - backbone/neck/检测头 ← `model_common/yolo26n.pt`（COCO 80 类；分类输出层因 nc 不同自动跳过）
  - 深度头 `dhead` ← `yolo26n-depth.pt` 的 `model.23.*`（Depth 头），两模型第 0–22 层结构相同，通道一致。

## 4. 损失

`DetDepthLoss`：

```
total = det_loss + dw * depth_loss
det_loss  = E2ELoss(model)（YOLO26 原检测损失，含 one2many/one2one 衰减，update() 透传）
depth_loss = DepthLoss26(model)(preds["depth"], batch)，dlam=1.0（完全尺度不变，适配相对伪深度）
```

- `loss_names = box, cls, dfl, dlog, dgrad`，训练日志同时显示两部分损失。
- `dw` 默认 1.0，作为训练脚本参数暴露（两者都按 batch size 缩放，量级相近；训练初期观察后再调）。

## 5. 数据与增强

- `DetDepthDataset(YOLODataset)`：正常读框标注，`get_image_and_label` 额外加载对应深度图并 resize 到图像尺寸
  （复用 `DepthDataset` 的路径映射与 `load_depth`）。
- 现有增强里 `RandomPerspective`、`RandomFlip`、`LetterBox` 已支持同步变换 `depth`；`Mosaic/MixUp/CutMix/CopyPaste`
  不支持深度。**第一版关闭 mosaic/mixup/cutmix/copy_paste**，保留仿射、翻转、HSV。
  - 代价：VisDrone 小目标检测 mAP 预计比开 mosaic 低几个点。后续若需要，再给 Mosaic 子类补深度拼接（作为二期）。
- Format 阶段：检测的 `Format` 之后把 `depth` 转成 (1,H,W) float tensor（复用 `DepthFormat.apply_depth` 的做法，
  但保留实例标注键）；`collate_fn` 已支持 stack `depth` 键。

## 6. 训练 / 验证

- `DetDepthTrainer(DetectionTrainer)`：覆盖 `build_dataset`、`get_model`（含上面两段权重加载）、
  `preprocess_batch`（depth 转 float）、`get_validator`、`loss_names`。task 仍为 `detect`。
- `DetDepthValidator(DetectionValidator)`：评价指标为检测 mAP（best.pt 按 mAP 选择）；深度质量通过验证损失
  `dlog/dgrad` 监控。不额外实现 AbsRel/δ1（伪标签上的绝对指标没有意义）。
- 默认超参：`imgsz=640, epochs=100, batch=16`，单卡 A6000。

## 7. 推理输出（`predict.py`）

- `DetDepthPredictor(DetectionPredictor)`：
  1. 检测部分走原 NMS/后处理，得到原图坐标的框。
  2. 深度图：先裁掉 letterbox 的 padding，再双线性插值到原图尺寸。
  3. 每个框取框内深度的**中位数**作为该目标深度；框完全无像素时取框中心点值。
  4. 返回 `Results(boxes=..., depth=深度图)`，每框深度存在 `results.box_depth`（tensor, (N,)）。
- 可视化：框标签显示为 `car 0.87 | d=12.3`（相对深度单位，不标 m）。
- 命令行：`python hck_workspace/detdepth/predict.py --weights best.pt --source <图片/目录>`，
  同时保存画框图和每图一个 txt（`cls conf x1 y1 x2 y2 depth`）。

## 验收标准

1. 伪标签：train/val 每张图都有对应 PNG；抽查可视化与教师推理一致。
2. 训练：能从两份预训练权重启动，损失 5 项都下降，无 NaN。
3. 检测：报告 val mAP50/mAP50-95。可选对照实验：同增强设置（无 mosaic）下的纯检测训练，联合模型 mAP50-95
   差距应在 1 个点以内（对照要额外一次完整训练，是否跑由用户决定）。
4. 深度：联合模型在 val 上框内深度与伪标签框内深度的 Spearman 相关 ≥ 0.9。
5. 推理脚本输出带深度的框图与 txt。

## 不在本期范围

- 米制深度标定（需飞行高度或真值）
- Mosaic 深度拼接
- 导出 ONNX/TensorRT
- 修改 `ultralytics/` 包、注册新 task
