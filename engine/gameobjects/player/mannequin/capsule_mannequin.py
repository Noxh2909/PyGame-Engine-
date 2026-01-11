import math
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniformMatrix4fv, GL_TRUE


class CapsuleMannequin:
    """
    Debug / Placeholder mannequin for the Player.
    No input, no physics, no state.
    """

    def __init__(self, player, body_mesh, head_mesh):
        self.player = player
        self.body_mesh = body_mesh
        self.head_mesh = head_mesh

    # ---------- helpers ----------

    def _yaw_matrix(self) -> NDArray[np.float32]:
        front = self.player.front
        yaw = math.atan2(front[0], front[2])

        c = math.cos(yaw)
        s = math.sin(yaw)

        return np.array(
            [
                [ c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _translate(self, pos: np.ndarray) -> NDArray[np.float32]:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        return T

    def _scale(self, s: float) -> NDArray[np.float32]:
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = s
        return S

    # ---------- model matrices ----------

    def body_model_matrix(self) -> NDArray[np.float32]:
        pos = self.player.position.copy().astype(np.float32)
        pos[1] += self.player.height * 0.5

        return (
            self._translate(pos)
            @ self._yaw_matrix()
            @ self._scale(0.5)
        ).astype(np.float32)

    def head_model_matrix(self) -> NDArray[np.float32]:
        pos = self.player.position.copy().astype(np.float32)
        pos[1] += self.player.height + 0.25

        return self._translate(pos) @ self._scale(0.25)

    # ---------- render ----------

    def draw(self, program):
        glUseProgram(program)

        model = self.body_model_matrix()
        loc = glGetUniformLocation(program, "u_model")
        glUniformMatrix4fv(loc, 1, GL_TRUE, model)
        self.body_mesh.draw()

        model = self.head_model_matrix()
        glUniformMatrix4fv(loc, 1, GL_TRUE, model)
        self.head_mesh.draw()
        
class CapsuleBodyMesh:
    def __init__(self, mesh):
        self.mesh = mesh

    def draw(self):
        self.mesh.draw()


class CapsuleHeadMesh:
    def __init__(self, mesh):
        self.mesh = mesh

    def draw(self):
        self.mesh.draw()