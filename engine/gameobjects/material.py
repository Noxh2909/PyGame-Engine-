class Material:
    def __init__(self, color=(1.0, 1.0, 1.0), texture: str | None = None):
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
        if name == "white":
            return Material(color=(0.0, 1.0, 1.0))

        raise ValueError(f"Unknown material: {name}")