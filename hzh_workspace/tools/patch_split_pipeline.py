#!/usr/bin/env python3
"""把 detdepth 流水线脚本复制到 hzh_workspace/tools/split_pipeline/ 并只改路径地址。"""
import pathlib
import shutil

REPO = pathlib.Path('/home/kang/workspace/projects/yolo-depth/ultralytics-depth')
SRC = REPO / 'hck_workspace/detdepth'
DST = REPO / 'hzh_workspace/tools/split_pipeline'
DST.mkdir(parents=True, exist_ok=True)

for name in ('export_onnx_split.py', 'convert_rknn.py', 'rknn_infer.py', 'run_rknn_pipeline.sh'):
    shutil.copy2(SRC / name, DST / name)

p = DST / 'export_onnx_split.py'
s = p.read_text()
s = s.replace('REPO = HERE.parents[1]', "REPO = Path('%s')" % REPO)
p.write_text(s)

p = DST / 'run_rknn_pipeline.sh'
s = p.read_text()
pairs = [
    ('PT=${1:-$REPO/hzh_workspace/runs/pfv_detdepth_from_scratch-3/weights/best.pt}',
     'PT=${1:-$REPO/hzh_workspace/runs/detdepth_coco_pf/weights/best.pt}  # v2 正式；v1 用 ..._old'),
    ('OUT=${2:-$REPO/hck_workspace/detdepth/export}',
     'OUT=${2:-$REPO/hzh_workspace/models/split_pf_v2}'),
    ('#   best.pt  = hzh_workspace/runs/pfv_detdepth_from_scratch-3/weights/best.pt',
     '#   best.pt  = hzh_workspace/runs/detdepth_coco_pf/weights/best.pt'),
    ('#   out_dir  = hck_workspace/detdepth/export',
     '#   out_dir  = hzh_workspace/models/split_pf_v2'),
    ('D=$REPO/hck_workspace/detdepth', 'D=$REPO/hzh_workspace/tools/split_pipeline'),
    ('PLATFORM=${PLATFORM:-rk3576}', 'PLATFORM=${PLATFORM:-rk3588}'),
    ('CALIB_DIR=/mnt/data2/PFV_dataset/images/train', 'CALIB_DIR=/mnt/data2/COCO_PF/images/train'),
    ('VAL_DIR=/mnt/data2/PFV_dataset/images/val', 'VAL_DIR=/mnt/data2/COCO_PF/images/val'),
]
for a, b in pairs:
    assert a in s, 'not found: ' + a
    s = s.replace(a, b)
p.write_text(s)
print('patched ->', DST)
