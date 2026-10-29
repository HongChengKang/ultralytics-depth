# -*- coding: utf-8 -*-
"""一次 step1，多组 float16 子串各 build 一个 rknn；自动剔除 RKNN 拒绝的层名。"""
import glob, os, re, sys
from rknn.api import RKNN

onnx, dataset, work = sys.argv[1], sys.argv[2], sys.argv[3]
variants = {}
for a in sys.argv[4:]:
    name, subs = a.split('=', 1)
    variants[name] = subs.split('|')

os.makedirs(work, exist_ok=True)
os.chdir(work)
for f in glob.glob('*.quantization.cfg') + glob.glob('*.model') + glob.glob('*.data'):
    os.remove(f)

rk = RKNN(verbose=False)
rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
assert rk.load_onnx(model=onnx) == 0
print('step1 ...', flush=True)
assert rk.hybrid_quantization_step1(dataset=dataset) == 0
cfg = glob.glob('*.quantization.cfg')[0]
model = glob.glob('*.model')[0]
data = glob.glob('*.data')[0]
base = open(cfg).read()
names = re.findall(r'^    (/\S+):\s*$', base, re.M)
print('cfg layers:', len(names), flush=True)

for name, subs in variants.items():
    hs = [n for n in names if any(s in n for s in subs)]
    for attempt in range(30):
        body = 'custom_quantize_layers:\n' + ''.join('    %s: float16\n' % n for n in hs)
        txt = base.replace('custom_quantize_layers: {}\n', body, 1)
        p = os.path.join(work, name + '.cfg')
        open(p, 'w').write(txt)
        r2 = RKNN(verbose=False)
        r2.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
        try:
            assert r2.hybrid_quantization_step2(model_input=model, data_input=data, model_quantization_cfg=p) == 0
        except Exception as e:
            m = re.search(r"Invalid operands name '([^']+)'", str(e))
            if not m:
                print('FAIL', name, str(e)[-400:], flush=True); break
            bad = m.group(1)
            if bad not in hs:
                print('FAIL(unknown)', name, bad, flush=True); break
            hs.remove(bad)
            print('dropped invalid layer:', bad, flush=True)
            continue
        out = os.path.join(work, name + '.rknn')
        assert r2.export_rknn(out) == 0
        r2.release()
        print('built %s float16_layers=%d size=%d' % (name, len(hs), os.path.getsize(out)), flush=True)
        break
rk.release()
