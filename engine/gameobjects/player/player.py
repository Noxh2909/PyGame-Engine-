import numpy as np

from gameobjects.transform import Transform
from gameobjects.collider.capsule import CapsuleCollider
from gameobjects.player.states import PlayerState
from gameobjects.player.movement import PlayerMovement


class Player:
    def __init__(self, position=(0, 2, 0)):
        # --- transform ---
        self.transform = Transform(position=np.array(position, dtype=np.float32))

        # --- collider (player body) ---
        self.collider = CapsuleCollider(
            radius=0.35,
            height=1.8
        )

        # --- movement state ---
        self.velocity = np.zeros(3, dtype=np.float32)
        self.on_ground = False
        self.state = PlayerState.IDLE

        # --- movement controller ---
        self.movement = PlayerMovement(self)

    def update(self, wish_dir, want_jump, want_sprint, dt: float):
        """
        Update player movement.
        All movement logic is delegated to PlayerMovement.
        """
        self.movement.update(
            wish_dir=wish_dir,
            want_jump=want_jump,
            want_sprint=want_sprint,
            dt=dt
        )