import json
import numpy as np
from typing import Dict, Any


class FBXJsonLoader:
    """
    Loader for FBX-exported JSON files.
    Each JSON file has ONE responsibility:
      - mesh.json       -> geometry
      - skin.json       -> joints + weights
      - skeleton.json   -> bone hierarchy + inverse bind
      - animation.json  -> animation clips
    """

    # ------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------
    def load_mesh(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = json.load(f)

        if "mesh" not in data:
            raise RuntimeError("Mesh JSON missing 'mesh' section")

        mesh = data["mesh"]

        if "vertices" not in mesh or "indices" not in mesh:
            raise RuntimeError("Mesh JSON must contain 'vertices' and 'indices'")

        vertices = np.array(mesh["vertices"], dtype=np.float32)
        indices = np.array(mesh["indices"], dtype=np.int32)

        return {
            "vertices": vertices,
            "indices": indices,
        }

    # ------------------------------------------------------------
    # Skin
    # ------------------------------------------------------------
    def load_skin(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = json.load(f)

        if "skin" not in data:
            raise RuntimeError("Skin JSON missing 'skin' section")

        skin = data["skin"]

        if "joints" not in skin or "weights" not in skin:
            raise RuntimeError("Skin JSON must contain 'joints' and 'weights'")

        joints = np.array(skin["joints"], dtype=np.int32)
        weights = np.array(skin["weights"], dtype=np.float32)

        if joints.ndim != 2 or joints.shape[1] != 4:
            raise RuntimeError(f"Invalid joints shape: {joints.shape}, expected (V,4)")

        if weights.ndim != 2 or weights.shape[1] != 4:
            raise RuntimeError(f"Invalid weights shape: {weights.shape}, expected (V,4)")

        return {
            "joints": joints,
            "weights": weights,
        }

    # ------------------------------------------------------------
    # Skeleton
    # ------------------------------------------------------------
    def load_skeleton(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = json.load(f)

        if "bones" not in data:
            raise RuntimeError("Skeleton JSON missing 'bones' section")

        parents = []
        names = []
        inverse_bind = []

        for b in data["bones"]:
            parents.append(b["parent"])
            names.append(b["name"])

            m = np.array(b["inverse_bind"], dtype=np.float32)
            inverse_bind.append(m.reshape(4, 4))

        return {
            "parents": parents,
            "names": names,
            "inverse_bind": inverse_bind,
        }

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    def load_animations(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = json.load(f)

        if "animations" not in data:
            raise RuntimeError("Animation JSON missing 'animations' section")

        return data["animations"]