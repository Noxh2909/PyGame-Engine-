import numpy as np


class AABBCollider:
    def __init__(self, size=(1, 1, 1)):
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
        # World-space collider size = visual scale × local collider size
        world_size = self.size * transform.scale

        # Small safety margin to prevent side-clipping (in meters)
        collision_margin = 0.8

        half = (world_size * 0.5) + collision_margin

        min_v = transform.position - half
        max_v = transform.position + half
        return min_v, max_v
