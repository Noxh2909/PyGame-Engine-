import math
import numpy as np
from numpy.typing import NDArray
from gameobjects.object import GameObject


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
        self.player = player
        self.body_mesh = body_mesh
        self.body_height = height

        super().__init__(
            mesh=body_mesh,
            transform=self,
            material=material,
        )

    # -------------------------------------------------
    # Static transform helpers (TPS / world mannequin)
    # -------------------------------------------------

    def _yaw_rotation(self) -> NDArray[np.float32]:
        if self.player is None:
            return np.eye(4, dtype=np.float32)

        front = self.player.front
        yaw = math.atan2(front[0], front[2])
        c = math.cos(yaw)
        s = math.sin(yaw)

        return np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _translation(self, pos: np.ndarray) -> NDArray[np.float32]:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        return T

    def _scale(self, s: float = 1.0) -> NDArray[np.float32]:
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = s
        return S

    def model_matrix(self) -> NDArray[np.float32]:
        if self.player is None:
            return np.eye(4, dtype=np.float32)

        pos = self.player.position.copy().astype(np.float32)
        pos[1] += self.body_height * -1.2

        return (
            self._translation(pos) @ self._yaw_rotation() @ self._scale(2.4)
        ).astype(np.float32)

    # GameObject transform interface
    def matrix(self) -> NDArray[np.float32]:
        return self.model_matrix()
