from PIL import Image
import io
from pygltflib import GLTF2
import numpy as np
import struct
from typing import Optional


class GLBLoader:
    def __init__(self, path: str):
        gltf = GLTF2().load(path)
        if gltf is None:
            raise ValueError(f"Failed to load GLTF: {path}")
        self.gltf: GLTF2 = gltf

        binary = self.gltf.binary_blob()
        if binary is None:
            raise ValueError("GLTF has no binary blob (use .glb)")
        self._binary: bytes = binary

    # -------------------------------------------------
    # Accessor helpers
    # -------------------------------------------------
    def _component_count(self, accessor_type: str) -> int:
        return {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[accessor_type]

    def _component_format(self, component_type: int):
        if component_type == 5126:  # FLOAT
            return "f", 4
        if component_type == 5125:  # UNSIGNED_INT
            return "I", 4
        if component_type == 5123:  # UNSIGNED_SHORT
            return "H", 2
        if component_type == 5121:  # UNSIGNED_BYTE
            return "B", 1
        raise ValueError(f"Unsupported component type: {component_type}")

    def _read_accessor(self, acc_index: int) -> np.ndarray:
        acc = self.gltf.accessors[acc_index]
        if acc.bufferView is None:
            raise ValueError("Accessor has no bufferView")

        view = self.gltf.bufferViews[acc.bufferView]
        comp_char, comp_size = self._component_format(acc.componentType)
        comps = self._component_count(acc.type)

        stride = view.byteStride or (comp_size * comps)
        offset = (view.byteOffset or 0) + (acc.byteOffset or 0)

        fmt = "<" + comp_char * comps
        out = np.empty((acc.count, comps), dtype=np.float32)

        buf: bytes = self._binary  # type narrowing for static checkers
        for i in range(acc.count):
            out[i] = struct.unpack_from(fmt, buf, offset + i * stride)

        return out

    # -------------------------------------------------
    # Animation helpers
    # -------------------------------------------------
    def _read_animation_sampler(self, sampler):
        if sampler.input is None or sampler.output is None:
            return None, None

        times = self._read_accessor(sampler.input).flatten()
        values = self._read_accessor(sampler.output)
        return times.astype(np.float32), values.astype(np.float32)

    # -------------------------------------------------
    # Texture helpers
    # -------------------------------------------------
    def _load_basecolor_texture(self, prim) -> Optional[Image.Image]:
        if prim.material is None:
            return None

        material = self.gltf.materials[prim.material]
        if material.pbrMetallicRoughness is None:
            return None

        tex_info = material.pbrMetallicRoughness.baseColorTexture
        if tex_info is None:
            return None

        texture = self.gltf.textures[tex_info.index]
        image = self.gltf.images[texture.source]

        if image.bufferView is None:
            raise ValueError("External textures (.gltf) not supported yet")

        view = self.gltf.bufferViews[image.bufferView]
        offset = view.byteOffset or 0
        length = view.byteLength

        img_bytes = self._binary[offset : offset + length]
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def load_first_mesh(self) -> dict:
        if not self.gltf.meshes:
            raise ValueError("No meshes found in GLTF")

        mesh = self.gltf.meshes[0]
        prim = mesh.primitives[0]

        if prim.attributes.POSITION is None:
            raise ValueError("GLTF primitive has no POSITION attribute")

        positions = self._read_accessor(prim.attributes.POSITION)

        normals = (
            self._read_accessor(prim.attributes.NORMAL)
            if prim.attributes.NORMAL is not None
            else np.zeros_like(positions)
        )

        uvs = (
            self._read_accessor(prim.attributes.TEXCOORD_0)
            if prim.attributes.TEXCOORD_0 is not None
            else np.zeros((positions.shape[0], 2), dtype=np.float32)
        )

        indices = None
        if prim.indices is not None:
            idx = self._read_accessor(prim.indices)
            indices = idx.astype(np.uint32).flatten()

        vertices = np.hstack([positions, normals, uvs]).astype(np.float32)
        albedo = self._load_basecolor_texture(prim)

        # animations
        animations = []
        if self.gltf.animations:
            for anim in self.gltf.animations:
                anim_data = {
                    "name": anim.name,
                    "samplers": [],
                    "channels": [],
                }

                for sampler in anim.samplers:
                    times, values = self._read_animation_sampler(sampler)
                    anim_data["samplers"].append(
                        {
                            "times": times,
                            "values": values,
                            "interpolation": sampler.interpolation or "LINEAR",
                        }
                    )

                for ch in anim.channels:
                    anim_data["channels"].append(
                        {
                            "sampler": ch.sampler,
                            "node": ch.target.node,
                            "path": ch.target.path,
                        }
                    )

                animations.append(anim_data)

        return {
            "vertices": vertices,
            "indices": indices,
            "albedo": albedo,
            "animations": animations,
            "nodes": self.gltf.nodes or [],
            "skins": self.gltf.skins or [],
        }
