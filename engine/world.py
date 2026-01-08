# world.py
import json
from pathlib import Path

from gameobjects.mesh import Mesh, MeshRegistry
from gameobjects.object import GameObject, Transform
from gameobjects.collider import AABBCollider
from gameobjects.material import MaterialRegistry

class World:
    def __init__(self, level_path: str | None = None):
        """
        Docstring für __init__
        
        :param self: The object itself
        """
        self.objects: list[GameObject] = []
        self.static_objects: list[GameObject] = []
        if level_path:
            self.load_level(level_path)
            
    def _create_object(self, data: dict):
        """
        Docstring für _create_object

        :param self: The object itself
        :param data: dict with object parameters
        :type data: dict
        """
        # Transform
        position = data.get("position", [0, 0, 0])
        scale = data.get("scale", [1, 1, 1])

        transform = Transform(position=position, scale=scale)

        # Mesh (optional)
        mesh = None
        if "mesh" in data:
            mesh = MeshRegistry.get(data["mesh"])
            
        material = None
        if "material" in data:
            material = MaterialRegistry.get(data["material"])

        # Collider (optional)
        collider = None
        if "collider" in data:
            collider = AABBCollider(size=data["collider"])

        # Color (optional)

        obj = GameObject(
            mesh=mesh,
            transform=transform,
            material=material,
            collider=collider
        )

        self.objects.append(obj)

        if collider is not None:
            self.static_objects.append(obj)

    def load_level(self, level_path: str):
        """
        Docstring für load_level
        
        :param self: The object itself
        :param level_path: Path to the level file
        :type level_path: str
        """
        path = Path(level_path)
        if not path.exists():
            raise FileNotFoundError(f"Level file not found: {level_path}")

        with open(path, "r") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        for entry in objects:
            self._create_object(entry)
