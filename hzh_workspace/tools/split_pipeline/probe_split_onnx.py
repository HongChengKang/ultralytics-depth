import sys
import numpy as np
import cv2
import onnxruntime as ort

onnx_path, img_path = sys.argv[1], sys.argv[2]
W = H = 640
im = cv2.imread(img_path)
h, w = im.shape[:2]
s = min(W / w, H / h)
rw, rh = int(w * s), int(h * s)
left = ((W - rw) // 2) // 4 * 4
top = ((H - rh) // 2) // 2 * 2
canvas = np.full((H, W, 3), 114, np.uint8)
canvas[top:top + rh, left:left + rw] = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
x = canvas[:, :, ::-1].astype(np.float32) / 255.0
x = np.ascontiguousarray(x.transpose(2, 0, 1))[None].repeat(4, 0)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
outs = sess.run(None, {sess.get_inputs()[0].name: x})
print("outputs:", names)
for li in range(3):
    cls = outs[3 * li + 1][0]
    box = outs[3 * li][0]
    print("level", li, "cls shape", cls.shape,
          "cls0 max %.4f" % cls[0].max(),
          "cls1 max %.4f" % cls[1].max() if cls.shape[0] > 1 else "",
          "| box max %.3f" % box.max(),
          "| score_sum max %.4f" % outs[3 * li + 2][0].max())
top_score = max(outs[3 * li + 1][0].max() for li in range(3))
print("overall cls max", top_score)
