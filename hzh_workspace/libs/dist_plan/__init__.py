"""Compatibility shim for checkpoints pickled before the detdist rename.

Old ``best.pt`` files store ``dist_plan.head.DetectDist`` / ``dist_plan.model.DistModel``
in their pickle.  These two modules re-export the classes under the old names so
such checkpoints still load; new code must import from ``detdist``.
"""
