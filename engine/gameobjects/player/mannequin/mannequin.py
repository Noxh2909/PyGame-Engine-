import math
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniformMatrix4fv, GL_TRUE


def quat_to_mat4(q):
    """
    Convert quaternion [x,y,z,w] to 4x4 rotation matrix.
    """
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (yy + zz)
    m[0, 1] = 2.0 * (xy - wz)
    m[0, 2] = 2.0 * (xz + wy)
    m[1, 0] = 2.0 * (xy + wz)
    m[1, 1] = 1.0 - 2.0 * (xx + zz)
    m[1, 2] = 2.0 * (yz - wx)
    m[2, 0] = 2.0 * (xz - wy)
    m[2, 1] = 2.0 * (yz + wx)
    m[2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


class Mannequin:
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
        nodes=None,
        skins=None,
        animations=None,
        material=None,
    ):
        self.player = player
        self.body_mesh = body_mesh
        self.body_height = height
        self.material = material

        # animation data (optional)
        self.nodes = nodes or []
        self.skins = skins or []
        self.animations = animations or []

        # animation state
        self.time = 0.0
        self.active_animation = 0 if self.animations else None

        # skeleton
        self.joint_nodes = []
        self.inverse_bind_matrices = []
        self.node_to_joint = {}

        if self.skins:
            skin = self.skins[0]
            self.joint_nodes = list(skin.joints)
            for j, node_idx in enumerate(self.joint_nodes):
                self.node_to_joint[node_idx] = j

        # bone matrices
        self.local_mats = []
        self.global_mats = []
        self.final_mats = []

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

    def model_matrix(self) -> NDArray[np.float32]:
        if self.player is None:
            return np.eye(4, dtype=np.float32)

        pos = self.player.position.copy().astype(np.float32)
        pos[1] += self.body_height * -1.2

        return (
            self._translation(pos)
            @ self._yaw_rotation()
            @ self._scale(2.4)
        ).astype(np.float32)

    # -------------------------------------------------
    # Update (animation sampling – minimal)
    # -------------------------------------------------

    def update(self, dt: float):
        if not self.joint_nodes:
            return

        self.time += dt

        joint_count = len(self.joint_nodes)
        self.local_mats = self._identity_mats(joint_count)
        self.global_mats = self._identity_mats(joint_count)
        self.final_mats = self._identity_mats(joint_count)

        # ---- minimal animation sampling (rotation only, no interpolation)
        if self.active_animation is not None:
            anim = self.animations[self.active_animation]

            for ch in anim.get("channels", []):
                if ch.get("path") != "rotation":
                    continue

                sampler = anim["samplers"][ch["sampler"]]
                times = sampler["times"]
                values = sampler["values"]  # quaternions

                if times is None or values is None or len(times) == 0:
                    continue

                # nearest keyframe
                idx = int(np.searchsorted(times, self.time % times[-1], side="right") - 1)
                idx = max(0, min(idx, len(times) - 1))

                node_idx = ch["node"]
                joint = self.node_to_joint.get(node_idx)
                if joint is None:
                    continue

                q = values[idx]
                self.local_mats[joint] = quat_to_mat4(q)

        # ---- hierarchy (parent -> child)
        for i, node_idx in enumerate(self.joint_nodes):
            parent = self.nodes[node_idx].parent if hasattr(self.nodes[node_idx], "parent") else None
            if parent is None or parent not in self.node_to_joint:
                self.global_mats[i] = self.local_mats[i]
            else:
                p = self.node_to_joint[parent]
                self.global_mats[i] = self.global_mats[p] @ self.local_mats[i]

        # ---- skinning (if inverse bind matrices provided)
        if self.inverse_bind_matrices:
            for i in range(joint_count):
                self.final_mats[i] = self.global_mats[i] @ self.inverse_bind_matrices[i]
        else:
            self.final_mats = self.global_mats

    def _identity_mats(self, n: int):
        return [np.eye(4, dtype=np.float32) for _ in range(n)]

    # -------------------------------------------------
    # Render
    # -------------------------------------------------

    def draw(self, program: int):
        """
        Draws the mannequin using the given shader program.
        Assumes view/projection are already set.
        """
        if self.body_mesh is None:
            return

        glUseProgram(program)

        model = self.model_matrix()
        loc = glGetUniformLocation(program, "u_model")
        glUniformMatrix4fv(loc, 1, GL_TRUE, model)

        self.body_mesh.draw()

    # -------------------------------------------------
    # Renderer compatibility (same as StaticMannequin)
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
