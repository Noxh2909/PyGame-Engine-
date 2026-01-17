import numpy as np

class Transform:
    def __init__(self, position=(0,0,0), rotation=(0.0,0.0,0.0), scale=(0,0,0)):
        """
        Docstring für __init__

        :param self: The object itself
        :param position: The position of the transform
        :param rotation: The rotation of the transform (Euler angles in radians)
        :param scale: The scale of the transform
        """
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale    = np.array(scale, dtype=np.float32)

    def matrix(self):
        """
        Docstring für matrix
        
        :param self: The object itself
        """
        m = np.identity(4, dtype=np.float32)

        # Scale
        m[0,0] = self.scale[0]
        m[1,1] = self.scale[1]
        m[2,2] = self.scale[2]

        # Rotation (Euler XYZ)
        rx, ry, rz = self.rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0, 0],
                       [0, cx, -sx, 0],
                       [0, sx, cx, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        Ry = np.array([[cy, 0, sy, 0],
                       [0, 1, 0, 0],
                       [-sy, 0, cy, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        Rz = np.array([[cz, -sz, 0, 0],
                       [sz, cz, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

        m = Rz @ Ry @ Rx @ m

        # Translation
        m[:3, 3] = self.position
        return m