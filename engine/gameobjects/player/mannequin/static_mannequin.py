import math
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniformMatrix4fv, GL_TRUE


class StaticMannequin:
    """
    Static (non-animated) mannequin.

    - Visual-only representation of the Player
    - No physics, no input, no state
    - Uses a real mesh (e.g. imported from FBX/OBJ)
    - Aligned to the Player's position and yaw
    """

    def __init__(self, player, body_mesh, height: float, material=None):
        self.player = player
        self.body_mesh = body_mesh
        self.body_height = height
        self.material = material

    # -------------------------------------------------
    # Transform helpers
    # -------------------------------------------------

    def _yaw_rotation(self) -> NDArray[np.float32]:
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
        
    def _translation(self, pos: np.ndarray) -> NDArray[np.float32]:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        return T

    def _scale(self, s: float = 1.0) -> NDArray[np.float32]:
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = s
        return S

    # -------------------------------------------------
    # Model matrix
    # -------------------------------------------------

    def model_matrix(self) -> NDArray[np.float32]:
        """
        Builds the model matrix so that the mannequin's feet
        are placed on the ground (y = 0).
        """
        pos = self.player.position.copy().astype(np.float32)
        pos[1] += self.body_height * -1.2  # ground alignment

        return (
            self._translation(pos)
            @ self._yaw_rotation()
            @ self._scale(2.4) #size of mannequin
        ).astype(np.float32)

    # -------------------------------------------------
    # Render
    # -------------------------------------------------

    def draw(self, program: int):
        """
        Draws the mannequin using the given shader program.
        Assumes view/projection are already set.
        """
        glUseProgram(program)

        model = self.model_matrix()
        loc = glGetUniformLocation(program, "u_model")
        glUniformMatrix4fv(loc, 1, GL_TRUE, model)

        self.body_mesh.draw()

    # -------------------------------------------------
    # Renderer compatibility
    # -------------------------------------------------

    @property
    def mesh(self):
        return self.body_mesh

    @property
    def transform(self):
        class _T:
            def __init__(self, m):
                self._m = m

            def matrix(self):
                return self._m

        return _T(self.model_matrix())
