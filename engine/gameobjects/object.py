import numpy as np
from typing import Optional
from gameobjects.collider import AABBCollider

class Transform:
    def __init__(self, position=(0,0,0), scale=(1,1,1)):
        """
        Docstring für __init__

        :param self: The object itself
        :param position: The position of the transform
        :param scale: The scale of the transform
        """
        self.position = np.array(position, dtype=np.float32)
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

        # Translation
        m[:3, 3] = self.position
        return m
    
class GameObject:
    def __init__(self, mesh, transform, color=(1.0,1.0,1.0), collider=None):
        """
        Docstring für __init__

        :param self: The object itself
        :param mesh: The mesh of the object
        :param transform: The transform of the object
        :param color: The color of the object
        :param collider: The collider of the object
        """
        self.mesh = mesh
        self.transform = transform
        self.color = color
        self.collider = collider