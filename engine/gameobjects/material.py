import os
from texture import load_texture


class Material:
    def __init__(self, color=(1.0, 1.0, 1.0), texture=None):
        """
        color   : fallback color (vec3)
        texture : OpenGL texture id or None
        """
        self.color = color
        self.texture = texture


class MaterialRegistry:
    _materials: dict[str, Material] = {}

    @classmethod
    def get(cls, name: str) -> Material:
        if name not in cls._materials:
            cls._materials[name] = cls._load(name)
        return cls._materials[name]

    @staticmethod
    def _load(name: str) -> Material:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        texture_dir = os.path.join(base_dir, "gameobjects", "assets", "textures")

        if name == "white_bricks":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "white_bricks.jpg")
                )
            )
        if name == "brick_wall":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "brick_wall.jpg")
                )
            )
        if name == "white":
            return Material(color=(1.0, 1.0, 1.0))

        raise ValueError(f"Unknown material: {name}")