import numpy as np
import math

from .player import look_at


class Camera:
    def __init__(self, player, fov=120.0):
        self.player = player
        self.fov = fov

    def get_view_matrix(self) -> np.ndarray:
        eye = self.player.position.copy()
        eye[1] += self.player.height + self.player._headbob_offset

        return look_at(
            eye,
            eye + self.player.front,
            self.player.up
        )

    def get_projection_matrix(self, aspect: float, near=0.1, far=100.0) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0

        return proj
