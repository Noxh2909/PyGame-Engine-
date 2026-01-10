from gameobjects.material import Material
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
    "sun": lambda: Material(color=(1,1,1), emissive=True)
}

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