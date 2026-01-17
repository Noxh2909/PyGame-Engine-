import numpy as np

class Skeleton:
    def __init__(self, parents, inverse_bind):
        assert parents is not None, "Skeleton parents must not be None"
        assert inverse_bind is not None, "Skeleton inverse_bind must not be None"

        self.parents = list(parents)
        self.count = len(self.parents)
        self.bone_count = self.count

        # validate inverse bind matrices
        assert len(inverse_bind) == self.count, (
            f"inverse_bind count {len(inverse_bind)} != parents count {self.count}"
        )

        self.inverse_bind = []
        for i, m in enumerate(inverse_bind):
            m = np.asarray(m, dtype=np.float32)
            assert m.shape == (4, 4), f"inverse_bind[{i}] must be 4x4"
            self.inverse_bind.append(m)

        self.local = [np.eye(4, dtype=np.float32) for _ in range(self.count)]
        self.world = [np.eye(4, dtype=np.float32) for _ in range(self.count)]
        self.final = [np.eye(4, dtype=np.float32) for _ in range(self.count)]

        # flat buffer for GPU upload (updated each frame)
        self.final_flat = np.ascontiguousarray(
            np.zeros((self.count, 4, 4), dtype=np.float32)
        )

        # renderer compatibility aliases
        self.final_mats_flat = self.final_flat.reshape(self.count * 16)

        # animation storage (name -> animation data)
        self.animations = {}

    def reset_pose(self):
        """
        Reset all local bone transforms to identity.
        Call once per frame BEFORE applying animation.
        """
        for i in range(self.count):
            self.local[i].fill(0.0)
            self.local[i][0, 0] = 1.0
            self.local[i][1, 1] = 1.0
            self.local[i][2, 2] = 1.0
            self.local[i][3, 3] = 1.0

    def update(self):
        """
        Build world and final (skinning) matrices.
        local -> world -> final = world * inverse_bind
        """
        for i in range(self.count):
            p = self.parents[i]
            if p >= 0:
                self.world[i][:] = self.world[p] @ self.local[i]
            else:
                self.world[i][:] = self.local[i]

            self.final[i][:] = self.world[i] @ self.inverse_bind[i]
            # transpose for OpenGL column-major layout
            self.final_flat[i][:] = self.final[i].T

        # contiguous flat buffer for GPU
        self.final_flat = np.ascontiguousarray(self.final_flat)
        self.final_mats_flat = self.final_flat.reshape(self.count * 16)