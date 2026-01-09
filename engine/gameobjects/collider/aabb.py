import numpy as np

class AABBCollider:
    def __init__(self, size=(1,1,1)):
        """
        Docstring für __init__

        :param self: The object itself
        :param size: The size of the collider
        """
        self.size = np.array(size, dtype=np.float32)

    def get_bounds(self, transform):
        """
        Docstring für get_bounds

        :param self: The object itself
        :param transform: The transform of the object
        """
        half = self.size * 0.5
        min_v = transform.position - half
        max_v = transform.position + half
        return min_v, max_v