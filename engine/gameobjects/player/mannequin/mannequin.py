import math
import numpy as np       
from typing import Optional
from numpy.typing import NDArray

from gameobjects.object import GameObject
from assets.animations.skeleton import Skeleton
from gameobjects.transform import Transform

class Mannequin(GameObject):
    """
    Animated mannequin.

    - Owns skeleton + animation state (optional)
    - Produces per-frame bone matrices
    - Renderer-agnostic (same interface as StaticMannequin)
    """

    def __init__(
        self,
        player=None,
        body_mesh=None,
        height: float = 0.0,
        material=None,
    ):
        # -------------------------
        # Runtime skinning attributes
        # -------------------------

        self.is_skinned: bool = False
        self.skeleton: Optional[Skeleton] = None

        self.player = player
        self.body_mesh = body_mesh
        # Visual body height (model-space, not camera / gameplay height)
        self.body_height = height

        super().__init__(
            mesh=body_mesh,
            transform=Transform(),
            material=material,
        )

        self.transform.scale[:] = (0.01, 0.01, 0.01)

        # FBX / Mixamo forward-axis correction
        # Engine expects +Z forward, model is rotated in source file
        self.yaw_sign: float = -1.0
        self.yaw_offset: float = math.radians(90.0)

    def update_from_player(self, player):
        """
        Synchronize mannequin with player transform.
        - Position = player feet position
        - Rotation = player yaw only (body follows camera yaw)
        """

        # Position: feet-on-ground
        self.transform.position[:] = player.transform.position

        # Rotation: yaw only (no pitch / roll)
        yaw = player.transform.rotation[1]
        self.transform.rotation[:] = (
            0.0,
            self.yaw_sign * yaw + self.yaw_offset,
            0.0,
        )

    def _translation(self, pos: np.ndarray) -> NDArray[np.float32]:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        return T

    def model_matrix(self) -> NDArray[np.float32]:
        # Body follows transform only (rotation handled externally)
        T = self._translation(self.transform.position)
        R = self.transform.rotation_matrix()
        S = self.transform.scale_matrix()
        return (T @ R @ S).astype(np.float32)
        
    # GameObject transform interface
    def matrix(self) -> NDArray[np.float32]:
        return self.transform.model_matrix()

    # @property
    # def bone_count(self) -> int:
    #     if self.skeleton is None:
    #         return 0
    #     return len(self.skeleton.bones)
