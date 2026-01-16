import json
import numpy as np

def load_skeleton_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    bones = data["skeleton"]["bones"]

    parents = []
    inv_bind = []
    names = []

    for b in bones:
        parents.append(b["parent"])
        names.append(b["name"])

        m = np.array(b["inverse_bind"], dtype=np.float32)
        inv_bind.append(m.reshape(4, 4))

    if "mesh" not in data:
        raise RuntimeError("JSON is missing required 'mesh' section")

    mesh = data["mesh"]

    if "joints" not in mesh or "weights" not in mesh:
        raise RuntimeError("JSON mesh must contain 'joints' and 'weights'")

    joints = np.array(mesh["joints"], dtype=np.int32)
    weights = np.array(mesh["weights"], dtype=np.float32)

    if joints.ndim != 2 or joints.shape[1] != 4:
        raise RuntimeError(f"Invalid joints shape: {joints.shape}, expected (V, 4)")

    if weights.ndim != 2 or weights.shape[1] != 4:
        raise RuntimeError(f"Invalid weights shape: {weights.shape}, expected (V, 4)")

    if len(joints) == 0:
        raise RuntimeError("Joints array is empty")

    if len(weights) == 0:
        raise RuntimeError("Weights array is empty")

    return {
        "names": names,
        "parents": parents,
        "inverse_bind": inv_bind,
        "joints": joints,
        "weights": weights,
    }