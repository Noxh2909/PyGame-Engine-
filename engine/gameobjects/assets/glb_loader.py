from PIL import Image
import io
from pygltflib import GLTF2
import numpy as np
import struct


def _component_count(accessor_type: str) -> int:
    return {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[accessor_type]


def _component_format(component_type: int):
    """
    Returns (struct_format_char, byte_size)
    glTF component types:
      5126 = FLOAT
      5125 = UNSIGNED_INT
      5123 = UNSIGNED_SHORT
      5121 = UNSIGNED_BYTE
    """
    if component_type == 5126:  # FLOAT
        return "f", 4
    if component_type == 5125:  # UNSIGNED_INT
        return "I", 4
    if component_type == 5123:  # UNSIGNED_SHORT
        return "H", 2
    if component_type == 5121:  # UNSIGNED_BYTE
        return "B", 1
    raise ValueError(f"Unsupported component type: {component_type}")


def _read_accessor(gltf: GLTF2, acc_index: int) -> np.ndarray:
    acc = gltf.accessors[acc_index]
    if acc.bufferView is None:
        raise ValueError("Accessor has no bufferView")
    view = gltf.bufferViews[acc.bufferView]
    data = gltf.binary_blob()
    if data is None:
        raise ValueError("GLTF has no binary blob (use .glb)")

    comp_char, comp_size = _component_format(acc.componentType)
    comps = _component_count(acc.type)

    stride = view.byteStride or (comp_size * comps)
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)

    fmt = "<" + comp_char * comps
    out = np.empty((acc.count, comps), dtype=np.float32)

    for i in range(acc.count):
        out[i] = struct.unpack_from(fmt, data, offset + i * stride)

    return out


# --- Texture Loading ---
def _load_basecolor_texture(gltf: GLTF2, prim) -> Image.Image | None:
    """
    Loads the baseColorTexture (albedo) for a primitive if present.
    Returns a PIL Image or None.
    """
    if prim.material is None:
        return None

    material = gltf.materials[prim.material]

    if material.pbrMetallicRoughness is None:
        return None

    tex_info = material.pbrMetallicRoughness.baseColorTexture
    if tex_info is None:
        return None

    texture = gltf.textures[tex_info.index]
    image = gltf.images[texture.source]

    # Only .glb supported here
    if image.bufferView is None:
        raise ValueError("External textures (.gltf) not supported yet")

    view = gltf.bufferViews[image.bufferView]
    data = gltf.binary_blob()

    offset = view.byteOffset or 0
    length = view.byteLength

    if data is not None: 
        img_bytes = data[offset : offset + length]
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


def load_gltf_mesh(path: str) -> tuple[np.ndarray, np.ndarray | None, Image.Image | None]:
    """
    Loads the FIRST mesh / FIRST primitive from a .glb file.
    Returns interleaved vertices: position (3), normal (3), uv (2).
    Also returns indices if present.

    Requirements:
    - glTF Binary (.glb)
    - POSITION attribute required
    - NORMAL / TEXCOORD_0 optional (filled with defaults if missing)
    """

    gltf = GLTF2().load(path)

    if gltf is None:
        raise ValueError(f"Failed to load GLTF: {path}")

    if not gltf.meshes:
        raise ValueError(f"No meshes found in GLTF: {path}")

    mesh = gltf.meshes[0]
    prim = mesh.primitives[0]

    if prim.attributes.POSITION is None:
        raise ValueError("GLTF primitive has no POSITION attribute")

    positions = _read_accessor(gltf, prim.attributes.POSITION)

    normals = (
        _read_accessor(gltf, prim.attributes.NORMAL)
        if prim.attributes.NORMAL is not None
        else np.zeros_like(positions)
    )

    uvs = (
        _read_accessor(gltf, prim.attributes.TEXCOORD_0)
        if prim.attributes.TEXCOORD_0 is not None
        else np.zeros((positions.shape[0], 2), dtype=np.float32)
    )

    # Indices (VERY IMPORTANT for glTF / .glb)
    indices = None
    if prim.indices is not None:
        acc = gltf.accessors[prim.indices]
        if acc.type != "SCALAR":
            raise ValueError("Index accessor must be SCALAR")
        idx = _read_accessor(gltf, prim.indices)
        indices = idx.astype(np.uint32).flatten()

    vertices = np.hstack([positions, normals, uvs]).astype(np.float32)
    albedo_image = _load_basecolor_texture(gltf, prim)
    
    return vertices, indices, albedo_image