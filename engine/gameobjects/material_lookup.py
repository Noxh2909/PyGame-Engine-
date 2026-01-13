from gameobjects.texture import load_texture
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
texture_dir = os.path.join(base_dir, "gameobjects", "assets", "textures")

MATERIAL_TABLE = {
    "white": lambda: Material(color=(1,1,1)),
    "wood": lambda: Material(texture=load_texture(os.path.join(texture_dir, "wood_wall.jpg"))),
    "marble": lambda: Material(texture=load_texture(os.path.join(texture_dir, "marble_floor.jpg"))),
    "ocean": lambda: Material(texture=load_texture(os.path.join(texture_dir, "ocean.jpg"))),
    "destiny": lambda: Material(texture=load_texture(os.path.join(texture_dir, "destiny2.jpeg"))),
    "yasu": lambda: Material(texture=load_texture(os.path.join(texture_dir, "yasu.jpeg"))),
    "js": lambda: Material(texture=load_texture(os.path.join(texture_dir, "js.jpeg" ))),
    "elian": lambda: Material(texture=load_texture(os.path.join(texture_dir, "elian.jpeg" ))),
    "sun": lambda: Material(color=(1,1,1), emissive=True),
}

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
        if name in MATERIAL_TABLE:
            return MATERIAL_TABLE[name]()
        else:
            raise ValueError(f"Unknown material: {name}")
