import json
import numpy as np
from pathlib import Path

from gameobjects.mesh import Mesh
from gameobjects.material_lookup import Material
from gameobjects.texture import Texture


class Skeleton:
    def __init__(self, parents, inverse_bind):
        self.parents = parents              # (B,)
        self.inverse_bind = inverse_bind    # (B,4,4)
        self.num_bones = len(parents)


# === Bone matrix computation (bind pose, CPU side) ===
def compute_bone_matrices(skeleton: 'Skeleton') -> np.ndarray:
    """
    Compute final bone matrices in bind pose.
    Returns: (B,4,4) float32 array
    """
    B = skeleton.num_bones

    # Global pose (identity for now)
    global_pose = np.zeros((B, 4, 4), dtype=np.float32)
    for i in range(B):
        global_pose[i] = np.eye(4, dtype=np.float32)

    # Apply hierarchy (parent * local)
    for i in range(B):
        p = skeleton.parents[i]
        if p >= 0:
            global_pose[i] = global_pose[p] @ global_pose[i]

    # Final skinning matrices
    final_mats = np.zeros_like(global_pose)
    for i in range(B):
        final_mats[i] = global_pose[i] @ skeleton.inverse_bind[i]

    return final_mats


class FBXAsset:
    def __init__(self, mesh, skeleton, textures):
        self.mesh = mesh
        self.skeleton = skeleton
        self.textures = textures
        self.bone_matrices = compute_bone_matrices(skeleton)


def load_fbx_asset(base_path: str) -> FBXAsset:
    base = Path(base_path)

    # Expect directory with fixed filenames: mesh.json, skel.json, tex.json
    if not base.is_dir():
        raise RuntimeError(
            "FBX JSON loader expects a directory containing "
            "mesh.json / skel.json / tex.json"
        )

    mesh_path = base / "mesh.json"
    skel_path = base / "skel.json"
    tex_path  = base / "tex.json"

    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if not skel_path.exists():
        raise FileNotFoundError(skel_path)

    # ------------------------------
    # Load mesh
    # ------------------------------
    with open(mesh_path, "r") as f:
        mesh_json = json.load(f)

    positions = np.asarray(mesh_json["positions"], dtype=np.float32)
    normals   = np.asarray(mesh_json["normals"], dtype=np.float32)
    uvs       = np.asarray(mesh_json["uvs"], dtype=np.float32)
    bone_ids  = np.asarray(mesh_json["bone_ids"], dtype=np.uint16)
    weights   = np.asarray(mesh_json["bone_weights"], dtype=np.float32)

    mesh = Mesh(
        positions=positions,
        normals=normals,
        uvs=uvs,
        bone_ids=bone_ids,
        bone_weights=weights,
    )

    # ------------------------------
    # Load skeleton
    # ------------------------------
    with open(skel_path, "r") as f:
        skel_json = json.load(f)

    bones = skel_json["bones"]
    num_bones = len(bones)

    parents = np.full(num_bones, -1, dtype=np.int32)
    inverse_bind = np.zeros((num_bones, 4, 4), dtype=np.float32)

    for i, b in enumerate(bones):
        parent = b["parent"]
        if parent >= 0:
            parents[i] = parent
        inverse_bind[i] = np.array(
            b["inverse_bind"], dtype=np.float32
        ).reshape(4, 4)

    skeleton = Skeleton(parents, inverse_bind)

    # ------------------------------
    # Load textures
    # ------------------------------
    textures = {}
    if tex_path.exists():
        with open(tex_path, "r") as f:
            textures = json.load(f)

    return FBXAsset(mesh=mesh, skeleton=skeleton, textures=textures)


def create_mannequin_from_fbx(base_path: str):
    asset = load_fbx_asset(base_path)

    material = Material(color=(1.0, 1.0, 1.0))
    albedo_path = "assets/skins/mannequin_albedo.png"

    if albedo_path:
        material.texture = Texture.load_texture(albedo_path)

    return asset.mesh, asset.skeleton, material

def update_bone_matrices(asset: FBXAsset):
    asset.bone_matrices = compute_bone_matrices(asset.skeleton)