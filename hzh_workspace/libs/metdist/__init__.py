"""metdist: detection + metric-distance regression (formal name of the detdist_v2 flow).

Self-contained merge of detdist (v1 data/loss base) and detdist_v2 (deep log-depth
towers, asymmetric quantile loss, order loss, background swap). Legacy module
aliases are registered so checkpoints pickled as ``detdist_v2.*`` still load.
"""

import sys

from . import augment, data, head, loss, model, trainer  # noqa: F401

_PKG = sys.modules[__name__]
sys.modules.setdefault("detdist_v2", _PKG)
for _sub in ("augment", "data", "head", "loss", "model", "trainer"):
    sys.modules.setdefault("detdist_v2." + _sub, sys.modules[__name__ + "." + _sub])
