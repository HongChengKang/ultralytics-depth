"""Scene-identity breaking augmentation for detdist_v2 (training only).

BackgroundSwap: replace the whole frame with a random donor image from the
dataset (any group, coco included), then paste the person crops back at their
original pixel locations. Person pixels (size/appearance cues) and distance
labels stay unchanged; the background-distance correlation is destroyed, so
the distance head cannot memorise "this scene -> these distances".

Boxes whose objects were removed by the swap (bags, non-pasted persons) are
dropped from supervision. Zero inference cost: never applied at val/predict.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class BackgroundSwap:
    def __init__(self, dataset, p: float = 0.5, max_persons: int = 3):
        self.ds = dataset
        self.p = p
        self.max_persons = max_persons

    def __call__(self, labels):
        if self.p <= 0 or random.random() > self.p:
            return labels
        img = labels["img"]
        H, W = img.shape[:2]
        inst = labels["instances"]
        cls = labels["cls"]
        dist = labels.get("dist")
        if len(inst) == 0 or dist is None:
            return labels
        bb = inst.bboxes  # normalized xyxy
        if hasattr(bb, "numpy"):
            bb = bb.numpy()
        px = np.stack([bb[:, 0] * W, bb[:, 1] * H, bb[:, 2] * W, bb[:, 3] * H], 1)
        cand = [
            i
            for i in range(len(inst))
            if int(cls[i]) == 0 and float(dist[i]) >= 0
            and (px[i, 2] - px[i, 0]) > 8 and (px[i, 3] - px[i, 1]) > 8
        ]
        if not cand:
            return labels
        random.shuffle(cand)
        cand = cand[: self.max_persons]

        donor = None
        for _ in range(8):
            j = random.randrange(len(self.ds.im_files))
            if self.ds.im_files[j] != labels.get("im_file"):
                donor = cv2.imread(self.ds.im_files[j])
                if donor is not None:
                    break
        if donor is None:
            return labels
        new_img = cv2.resize(donor, (W, H), interpolation=cv2.INTER_LINEAR)
        for i in cand:
            x1, y1, x2, y2 = [int(round(v)) for v in px[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            new_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        labels["img"] = new_img
        labels["instances"] = inst[cand]
        labels["cls"] = cls[cand]
        labels["dist"] = dist[cand]
        return labels