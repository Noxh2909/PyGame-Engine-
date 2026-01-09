# world.py
import json
from pathlib import Path

from gameobjects.mesh import Mesh, MeshRegistry
from gameobjects.object import GameObject
from gameobjects.transform import Transform
from gameobjects.collider.aabb import AABBCollider
from gameobjects.material import MaterialRegistry

class World:
    def __init__(self, level_path: str | None = None):
        """
        Docstring für __init__
        
        :param self: The object itself
        """
        self.objects: list[GameObject] = []
        self.static_objects: list[GameObject] = []
        self.lights: list[GameObject] = []
        self.sun: GameObject | None = None
        if level_path:
            self.load_level(level_path)
            
    def _create_object(self, data: dict):
        """
        Docstring für _create_object

        :param self: The object itself
        :param data: dict with object parameters
        :type data: dict
        """
         # ---------- transform ----------
        position = data.get("position", [0, 0, 0])
        scale = data.get("scale", [1, 1, 1])
        transform = Transform(position=position, scale=scale)

        # ---------- mesh ----------
        mesh = None
        mesh_name = data.get("mesh")
        if mesh_name:
            mesh = MeshRegistry.get(mesh_name)

        # ---------- material ----------
        material = None
        material_name = data.get("material")
        if material_name:
            material = MaterialRegistry.get(material_name)

        # ---------- collider ----------
        collider = None
        collider_size = data.get("collider")
        if collider_size:
            collider = AABBCollider(size=collider_size)

        # ---------- light ----------
        light = data.get("light")

        obj = GameObject(
            mesh=mesh,
            transform=transform,
            material=material,
            collider=collider,
            light=light
        )

        self.objects.append(obj)

        # Light-emitting objects (can be multiple)
        if light is not None:
            self.lights.append(obj)
            if self.sun is None:
                self.sun = obj

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
