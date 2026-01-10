import os
from gameobjects.texture import load_texture


class Material:
    def __init__(self, color=(1.0, 1.0, 1.0), texture=None, emissive=False, texture_scale_mode=None, texture_scale_value=None):
        """
        color         : fallback color (vec3)
        texture       : OpenGL texture id or None
        emissive      : bool
        texture_scale_mode : optional scale mode for texture coordinates
        texture_scale_value : optional scale value for texture coordinates
        """
        self.color = color
        self.texture = texture
        self.emissive = emissive
        self.texture_scale_mode = texture_scale_mode
        self.texture_scale_value = texture_scale_value


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
        if name == "white":
            return Material(color=(1.0, 1.0, 1.0))

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
        if name == "metal":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "metal.png")
                )
            )
        if name == "marble":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "marble_floor.jpg")
                )
            )
        if name == "wood":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "wood_wall.jpg")
                )
            )
        if name == "ocean":
            return Material(
                texture=load_texture(
                    os.path.join(texture_dir, "ocean.jpg")
                )
            )
        if name == "sun":
            return Material(color=(1.0, 1.0, 1.0), emissive=True)

        raise ValueError(f"Unknown material: {name}")