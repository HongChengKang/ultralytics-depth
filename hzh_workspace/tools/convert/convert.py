# -*- coding: utf-8 -*-
"""
onnx -> int8 rknn（rknn-toolkit2 2.3.2）

结构照抄 rknn_model_zoo/examples/yolov8_seg/python/convert.py，只改了三处：
  1. DATASET_PATH 指向 VisDrone 的 test 图片目录（原版是一个 txt）
  2. 目录时自动等间隔抽 CALIB_NUM 张生成 dataset.txt，再交给 rknn.build()
  3. DEFAULT_RKNN_PATH 指到 hzh_workspace/weights/

用法（必须在 rknn 环境）：
    conda activate /home/eav-nanjin/miniforge3/envs/rknn
    python hzh_workspace/tools/convert/convert.py <onnx> [platform] [dtype] [output]
    python hzh_workspace/tools/convert/convert.py hzh_workspace/weights/best.onnx rk3588 i8
"""
import os
import sys

DATASET_PATH = '/mnt/data2/VisDrone数据集/images/test'
DEFAULT_RKNN_PATH = '/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace/weights/best_int8.rknn'
DEFAULT_QUANT = True

# 校准图片数量（等间隔抽样，避免 1610 张全跑太慢）
CALIB_NUM = 300
# 由图片目录生成的 dataset.txt
DATASET_TXT = '/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hzh_workspace/weights/dataset.txt'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

from rknn.api import RKNN


def build_dataset_list():
    """DATASET_PATH 是目录时，等间隔抽 CALIB_NUM 张写 dataset.txt 并返回其路径；是 txt 则原样返回。"""
    if os.path.isfile(DATASET_PATH):
        return DATASET_PATH
    images = sorted(
        os.path.join(DATASET_PATH, f)
        for f in os.listdir(DATASET_PATH)
        if f.lower().endswith(IMG_EXTS)
    )
    if not images:
        print('ERROR: no image found in {}'.format(DATASET_PATH))
        exit(1)
    n = min(CALIB_NUM, len(images))
    idx = [int(i * len(images) / n) for i in range(n)]  # 等间隔抽样
    with open(DATASET_TXT, 'w') as f:
        for i in idx:
            f.write(images[i] + '\n')
    print('dataset: {} images -> {} (total {})'.format(n, DATASET_TXT, len(images)))
    return DATASET_TXT


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print(" platform choose from [rk3562,rk3566,rk3568,rk3576,rk3588,rk1808,rv1109,rv1126]")
        print(" dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3576,rk3588,rk1808,rv1109,rv1126]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]
    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    dataset = build_dataset_list() if do_quant else None
    ret = rknn.build(do_quantization=do_quant, dataset=dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done  ->  {}'.format(output_path))

    # Release
    rknn.release()
