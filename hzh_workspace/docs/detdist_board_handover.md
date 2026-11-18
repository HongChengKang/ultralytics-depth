# detdist 板端（RK3588）推理流程交接文档

本文只讲**板端这一段**：拿到 `*_split_b1_3588_{int8,fp16}.rknn`（12 输出）之后，怎么用 C++ 把
「读图 → letterbox → 推理 → 解码/NMS/距离 → 评测/测速」跑起来，以及 A1/A2/A3 三项提速做了什么、
实测多少、哪些路走不通。训练侧见 `docs/detdist.md`。

---

## 0. 一句话

12 输出的 detdist int8 模型，在 RK3588 上 **NPU 47.0 → 38.8 ms、端到端 111.9 → 42.1 ms**
（图已在内存口径），精度与板端 Python 基线逐框一致（190/190 匹配，mean IoU 0.9919，距离 mean|Δ| 0.0005 m）。

---

## 1. 交付物

板端（`new3588@172.18.110.75`）：

```
~/hzh_ws/detdist/                 # 新工程，独立于 ~/hzh_ws/yolo-det/（旧流程一行未改）
  CMakeLists.txt
  src/
    main.cpp          入口：eval 模式 / --bench 模式，参数解析，per-image 循环
    engine.h/.cpp     rknn 生命周期 + 零拷贝输入输出 + 预处理调度 + 计时
    rga_pre.h/.cpp    letterbox 参数 + RGA 实现（保留）+ OpenCV 实现（默认）
    postprocess.h/.cpp 12 输出解码、score_sum 预筛、逐类 NMS、框内距离中位数、坐标映射
    metrics.h/.cpp    标签解析、mAP50/mAP50-95（101 点）、距离分桶、JSON/preds 输出
  include/rknpu2/     # 从 yolo-det 复制
  lib/librknnrt.so    # 从 yolo-det 复制
  models/dist_pf/{dist_pf_all_split_b1_3588_int8.rknn, dist_fp16.rknn}
  datasets -> ../yolo-det/datasets
  results/eval_{...}.json + .preds.txt
  rga_test.cpp / rga_test2.cpp   # RGA 排查用的小程序（证据保留）
```

服务器侧（生成模型/ONNX 的工具）：`hzh_workspace/tools/split_pipeline_detdist/`。

---

## 2. 数据流（逐段对应到代码）

```
imread (main.cpp:208)                      ← 现在最大开销，40.3 ms/图
  └─ letterbox_params (rga_pre.cpp)        ← 只算参数：scale/min、rw%4、rh%2、left%4、top%2
      └─ preprocess_bgr_to_rgb             ← 默认 OpenCV：resize 到 pad 后 ROI + cvtColor(BGR2RGB)
          └─ 写 in_mem->virt_addr          ← 零拷贝：直接写 rknn DMA buffer
              └─ rknn_run                  ← NPU，36.8~38.8 ms
                  └─ outs_[0..11]          ← rknn_set_io_mem 绑定的输出内存
                      └─ decode_dets       ← 预筛→解码→逐类 NMS→框内距离中位数
                          └─ map_back_inplace  ← 减 pad、除 scale，回到原图坐标
                              └─ evaluate     ← mAP / 距离分桶 / 延迟统计
```

`--bench N` 走同一 `infer` + `decode_dets`，只是换成固定图循环并输出 p50/p95。

---

## 3. 12 输出语义与解码（最容易错的地方）

输出顺序（`export_onnx_split.py` 固定）：

```
idx 0..2   box_p3  cls_p3  score_p3
idx 3..5   box_p4  cls_p4  score_p4
idx 6..8   box_p5  cls_p5  score_p5
idx 9..11  dist_p3 dist_p4 dist_p5
```

- 全部 `dims=[1,C,H,W]`、**NCHW**、`fmt=0`；int8 模型 `type=INT8`，需 `(q - zp) * scale` 反量化；
  fp16 用 half→float。板端实测 int8 的 `zp≈-117..-128`、`scale≈0.0033..0.058`。
- `box` 是 **ltrb**，单位是该级 stride（8/16/32）；解码：

```
s    = model_w / W_level
cx   = (x + 0.5) * s ;  cy = (y + 0.5) * s
x1   = cx - l*s ; y1 = cy - t*s ; x2 = cx + r*s ; y2 = cy + b*s    # 画布坐标
```

- `cls` 已 sigmoid，逐 anchor 在 nc 维取 argmax；`score_sum = cls.sum(1).clamp(0,1)`，是逐 anchor 的
  置信度上界，用它做预筛（`score_sum < conf` 直接跳过，连 argmax 都不做）。
- `dist_p*` 已是 **米**：`sigmoid(raw) * 15`，导出时已乘好，板端只反量化。
- 框级距离：对**最终保留的框**，按各级 stride 遍历 anchor 中心，中心落在框内才取值，三级合并取中位数：
  `xa=floor(x1/s-0.5) … xb=ceil(x2/s-0.5)`，`cy=(y+0.5)*s` 再判 `cy∈[y1,y2]`。
  注意框和 anchor 中心都在**画布坐标**里比较（早期版本按特征图尺寸裁剪，是 bug）。

---

## 4. letterbox 规则（必须与训练/Python 侧一致）

```
s   = min(640/w, 640/h)
rw  = int(w*s); rw -= rw % 4      # 宽 4 对齐
rh  = int(h*s); rh -= rh % 2      # 高 2 对齐
left = (640-rw)//2; left -= left % 4
top  = (640-rh)//2; top  -= top % 2
画布 640×640，填充值 114；图像贴到 (left, top)，尺寸 (rw, rh)
```

映射回原图时用**实际** `scale_x=rw/w、scale_y=rh/h`（代码里分开存），比 Python 侧统一用 `s` 更准一点点，
但两者差异 <0.5 像素，对齐验证里看不出来。

---

## 5. 零拷贝（A2）

```cpp
// init
in_attr_.type = RKNN_TENSOR_UINT8; in_attr_.fmt = RKNN_TENSOR_NHWC; in_attr_.pass_through = 0;
in_mem_ = rknn_create_mem(ctx_, in_size_);
rknn_set_io_mem(ctx_, in_mem_, &in_attr_);
for each output: m = rknn_create_mem(ctx_, attr.size); rknn_set_io_mem(ctx_, m, &attr);

// infer
preprocess(..., (uint8_t*)in_mem_->virt_addr, in_mem_->fd, ...);   // 直接写 DMA buffer
rknn_run(ctx_, nullptr);
out[i].data = out_mem[i]->virt_addr;
```

- 预处理能拿到 `fd` 就给 RGA 用，拿不到就走 CPU 写虚拟地址（同一块内存）。
- 输出也走 `set_io_mem`；只有 `DETDIST_NO_ZC=1` 时才回退 `rknn_inputs_set` + `rknn_outputs_get`。
- 回退路径必须用 `outs_held_` 标志防止 `rknn_outputs_release` 被调两次（踩过 double free）。
- 实测 A2：NPU **42.70 → 38.80 ms**（e2e 46.24 → 42.08），省 4 ms。

---

## 6. 后处理（A3）

1. **预筛**：`score_sum < conf` → 跳过（三层 8400 个 anchor，绝大多数在这一步被扔掉）。
2. **解码**：cls argmax（只在候选上做）、ltrb→框、裁到画布、丢弃 <1px 的退化框。
3. **NMS**：按 score 降序，逐类贪心，`IoU > iou` 抑制，`--max-det` 300。
4. **距离**：只对 NMS 后的框算三级中位数（不再对每个候选算一遍）。
5. **映射**：`(x - pad) / scale` 再裁到原图。

实测 post **0.53 ms**（640×480）/ 0.44 ms（2560×1440）。Python 侧这一段包含在“差值 56–65 ms”里。

---

## 7. 评测部分

- **mAP**：101 点插值，逐类、IoU 0.5:0.05:0.95；`ap_101` 与 ultralytics 口径一致。
- **距离**：GT 与预测按同类、IoU≥0.5 做 1:1 贪心匹配；`gt_m = d_norm*15`；
  输出 `n / MAE / bias(有符号) / median|err| / Spearman(排名实现，不依赖 scipy)`，
  并按 GT 距离分 0–5 / 5–10 / 10–15 m 三桶给 `n/gt_median/pred_median/mae/bias`。
- **标签**：支持 5 列（无距离）或 6 列（第 6 列 `d_norm`）；`-1` 视为无距离、不进距离指标。
- **JSON**：`results/eval_*.json`，字段见板端结果文件；`--preds` 输出逐框文本，便于与 Python 对齐。

---

## 8. 测速

`--bench N --warmup M`：固定一张图循环，分别计时

| 段 | 起点 | 终点 |
|---|---|---|
| pre | `letterbox_params` 前 | 预处理写完 DMA buffer |
| npu | 预处理后 | `rknn_run` 返回 |
| post | 推理后 | 解码+NMS+距离+映射完 |
| e2e | 段首 | 段尾（= pre+npu+post，不含 imread） |

输出 p50 / p95 / mean。A/B 开关：`DETDIST_NO_ZC=1`（A2）、`DETDIST_TRY_RGA=1`（A1 的 RGA 分支）、
`--no-rga`、`DETDIST_DEBUG=1`（打印每路输出范围/候选数）、`DETDIST_DUMP=1`（打印 NMS 前候选）。

---

## 9. 三项优化的实际结论

| 项 | 计划 | 实际 |
|---|---|---|
| A1 RGA 预处理 | RGA 一次做 crop→resize→pad + RGB 直出，−10~15 ms | **RGA 在本板不可用**（见 §10），改 OpenCV SIMD：pre **20 → 2.5 ms** |
| A2 零拷贝 | `rknn_set_io_mem` + NHWC uint8，−5~10 ms | 实现并生效：NPU **42.7 → 38.8 ms**，−4 ms |
| A3 后处理 | score_sum 预筛 + 逐类 NMS + 只在最终框算距离，−8~15 ms | post **0.5 ms**；相对 Python 解码段省 ~20 ms |

---

## 10. RGA 为什么没用上（排查记录）

```bash
# 1) 内核与用户态版本
dmesg | grep -i rga
#   rga: rga2 driver loaded successfully ver:3.2.63318
#   rga: Module initialized. v1.2.15
cat /sys/module/rga/version        # 不存在

# 2) librga 启动即报
#   librga fail to get driver version! Compatibility mode will be enabled.
ls -l /usr/lib/aarch64-linux-gnu/librga.so*     # 2.1.0

# 3) 最小复现（虚拟地址 / rknn dma-buf fd 两种都试）
./rga_test  <img>      # imfill: -1, improcess bgr->rgb: -1, rgb->rgb: -1
./rga_test2 <img> <rknn>  # fd->fd 同样 -1
# 4) 权限
ls -l /dev/dma_heap/*              # crw------- root root（非 root 不可用）
```

现象统一是 `NOT_SUPPORTED (-1)`，且 `IM_STATUS_SUCCESS == 1` 不是 0——判断成功要用枚举比较。
结论：**内核 RGA 驱动 1.2.15 与 librga 2.1.0 不匹配 + dma_heap 仅 root**，当前镜像上 RGA 路线无解；
要么换带新驱动（≥1.3）的镜像，要么以 root 运行并确认 dma_heap 权限。代码里 RGA 实现完整保留，
`DETDIST_TRY_RGA=1` 可随时在新镜像上复测，失败自动回退。

---

## 11. 精度对齐（三步法，可复用）

1. **确认布局**：打印 `rknn_tensor_attr`（dims/type/fmt/size/zp/scale）。本模型是 NCHW；曾试 NHWC 索引 → 预测数变 0，立即排除。
2. **确认反量化**：同一张图，用 Python `rknnlite` 取 `outs[3]` 扁平前几个值，与 C++ `dequant_at` 对比（0.899/1.465/1.908 一致）。
3. **确认通道序**：C++ 预测数只有 Python 的 2/3、mAP 0.18 → 检查预处理发现 `cpu_letterbox` 把 BGR 直接当 RGB 写；改成 `sc_idx = 2 - c` 后 **182 框 / mAP50 0.6977**，与 Python 完全一致。

最终逐框对齐（int8，95 图）：190/190 匹配，mean IoU **0.9919**（min 0.691，98.9% ≥0.9），
距离 `mean|Δ| 0.0005 m`（max 0.053）。

---

## 12. 实测数据（同一块板、val100 95 图、int8）

| 版本 | 读图 | pre | NPU | post | 端到端 | 差值 |
|---|---|---|---|---|---|---|
| 旧 detdepth（10 输出，Python） | — | — | 114.5 | — | 176.2 | 61.7 |
| detdist（12 输出，Python，原表） | — | — | 47.0 | — | 111.9 | 64.9 |
| detdist（12 输出，Python，本次复测） | ~30 | — | 50.2 | — | 106.7 | 56.5 |
| **detdist（12 输出，C++ 本流程）** | **40.3** | **5.2** | **36.8** | **0.4** | **82.8** | **46.0** |
| detdist（C++，图在内存，`--bench` p50） | — | 2.5 | 38.8 | 0.5 | **42.1** | **3.0** |

精度（int8）：190 框、mAP50 **0.7036**、mAP50-95 0.4673；距离 n=123、MAE 1.585 m、bias +1.055、
median|err| 1.214、ρ 0.567；分桶 0–5 m（n=94）2.738→4.250、5–10 m（n=21）6.667→6.802、
10–15 m（n=8）11.970→10.059。
fp16：182 框、mAP50 0.6976、MAE 1.531、ρ 0.593。

现在瓶颈是 `imread`（40.3 ms/图，60 jpg + 35 png，含 2560×1440 大图），已超过 NPU。

---

## 13. 从零部署

```bash
# 1) 拷依赖（从旧工程，不改旧工程）
mkdir -p ~/hzh_ws/detdist/{include,lib,models/dist_pf}
cp -r ~/hzh_ws/yolo-det/include/rknpu2 ~/hzh_ws/detdist/include/
cp    ~/hzh_ws/yolo-det/lib/librknnrt.so  ~/hzh_ws/detdist/lib/
cp    ~/hzh_ws/yolo-det/models/dist_pf/*.rknn ~/hzh_ws/detdist/models/dist_pf/
ln -sfn ~/hzh_ws/yolo-det/datasets ~/hzh_ws/detdist/datasets

# 2) 6 列标签目录（距离指标用；val100 里有 14 张属于 train split，所以要合并）
mkdir -p ~/hzh_ws/yolo-det/datasets/coco_pf_val100/labels6
find /mnt/.../labels_dist/{train,val} -name '*.txt' -exec ln -sf {} labels6/ \;

# 3) 构建
cd ~/hzh_ws/detdist
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4

# 4) 跑通
./build/detdist_eval --model models/dist_pf/dist_pf_all_split_b1_3588_int8.rknn \
  --images datasets/coco_pf_val100/images --labels datasets/coco_pf_val100/labels6 \
  --json results/eval_int8.json --preds results/eval_int8.preds.txt
```

工具链：g++ 9.4 / cmake 3.16 / OpenCV 4.2（头在 `/usr/include/opencv4`）/ librga 2.1.0 / librknnrt 2.3.0。
CMake 里链接 `librknnrt.so`、`rga`、OpenCV、pthread；C++14（`std::filesystem` 不可用，列图用 `dirent.h`）。

---

## 14. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `models/dist_pf/dist_pf_all_split_b1_3588_int8.rknn` | rknn 路径 |
| `--images` | 必填 | 图片目录（jpg/jpeg/png/bmp） |
| `--labels` | `<images>/../labels` | 5 或 6 列标签目录 |
| `--names` | `person,Fertilizer bag` | 类别名 |
| `--conf` / `--iou` | 0.25 / 0.7 | 预筛与 NMS 阈值 |
| `--core` | 0 | 0/1/2 单核，3 = 三核 |
| `--bench N` / `--warmup M` | 0 / 10 | 测速模式 |
| `--limit N` | 0 | 只跑前 N 张 |
| `--json` / `--preds` | 空 | 输出报告 / 逐框 |
| `--no-rga` | off | 强制 CPU 预处理 |

环境变量：`DETDIST_NO_ZC=1`、`DETDIST_TRY_RGA=1`、`DETDIST_DEBUG=1`、`DETDIST_DUMP=1`。

---

## 15. 踩过的坑（每条都是“现象 → 原因 → 修法”）

1. `exit=139` 段错误 → `cpu_letterbox` 里 `src[((y*stride)+x)*3+c]` 把 stride 又乘 3 → 改行指针。
2. 预测框只有 Python 的 2/3、mAP 0.18 → 预处理没做 BGR→RGB → `sc_idx = 2-c`。
3. 输出按 NHWC 取 → 预测 0 个 → 确认 `fmt=0` 即 NCHW。
4. 零拷贝关掉后 double free → `rknn_outputs_release` 调了两次 → `outs_held_` 标志。
5. C++14 + `std::filesystem` 链接失败 → 改 `dirent.h`。
6. `--labels` 默认算错一层 → `dir_of(images)` 而不是 `dir_of(dir_of(images))`。
7. 想用 5 列标签算距离 → 必须 6 列目录（且合并 train/val，因为 val100 混了 14 张 train）。
8. int8 下 `dist_p*` 可能输出整型 → 必须按 `zp/scale` 反量化，不能直接当米。
9. RGA 的 `IM_STATUS_SUCCESS` 是 1 不是 0 → 别用 `== 0` 判成功。
10. `rknn_create_mem` 的输入 buffer 用 `attr.size`（不是 w*h*c 硬算），否则 set_io_mem 失败。

---

## 16. 后续可做的

1. **读图**（当前 40 ms，最大头）：JPEG 硬解、NV12 直出、或让上游把解码后的图送进来；也可双缓冲流水把读图和 NPU 重叠。
2. **RGA**：换 ≥1.3 驱动的镜像，或 root + dma_heap，再用 `DETDIST_TRY_RGA=1` 复测，预期 pre 再降 1–2 ms。
3. **多核/多路**：`--core 3` + 每核一个 context，吞吐还能上一个台阶（单路延迟不变）。
4. **int8 距离精度**：dist 三路可尝试 fp16 混合精度；校准集按距离分层。
5. **后处理**：目前已是 0.5 ms，收益有限，不建议再投入。

---

## 17. 复现命令

```bash
cd ~/hzh_ws/detdist

# 精度（mAP + 距离分桶 + 延迟）
./build/detdist_eval --model models/dist_pf/dist_pf_all_split_b1_3588_int8.rknn \
  --images datasets/coco_pf_val100/images --labels datasets/coco_pf_val100/labels6 \
  --json results/eval_int8.json --preds results/eval_int8.preds.txt

# 测速 p50/p95
./build/detdist_eval --model models/dist_pf/dist_pf_all_split_b1_3588_int8.rknn \
  --images datasets/coco_pf_val100/images --bench 200 --warmup 20

# A2 对比
DETDIST_NO_ZC=1 ./build/detdist_eval --model ... --images ... --bench 150 --warmup 20
```
