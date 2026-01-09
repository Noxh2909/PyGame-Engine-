import numpy as np
from gameobjects.player.states import PlayerState


class PlayerMovement:
    """
    Complete player movement controller:
    - walk / sprint
    - air control
    - jump
    - gravity
    """

    def __init__(self, player):
        self.player = player

        # -------- tuning --------
        self.walk_speed = 4.0
        self.sprint_speed = 7.0

        self.acceleration = 30.0
        self.deceleration = 20.0
        self.air_control = 0.35

        self.jump_force = 6.5
        self.gravity = 18.0
        self.max_fall_speed = 30.0

    # --------------------------------------------------
    # update (called every frame)
    # --------------------------------------------------

    def update(self, wish_dir: np.ndarray, want_jump: bool, want_sprint: bool, dt: float):
        """
        wish_dir: desired horizontal movement direction (x,z)
        want_jump: jump input (bool)
        want_sprint: sprint input (bool)
        dt: delta time
        """

        # normalize input direction
        if np.linalg.norm(wish_dir) > 0.0:
            wish_dir = wish_dir / np.linalg.norm(wish_dir)

        vel = self.player.velocity

        # ---------------- gravity ----------------
        if not self.player.on_ground:
            vel[1] -= self.gravity * dt
            vel[1] = max(vel[1], -self.max_fall_speed)

        # ---------------- jump ----------------
        if want_jump and self.player.on_ground:
            vel[1] = self.jump_force
            self.player.on_ground = False
            self.player.state = PlayerState.JUMP

        # ---------------- speed selection ----------------
        speed = self.sprint_speed if want_sprint and self.player.on_ground else self.walk_speed

        # ---------------- horizontal movement ----------------
        if self.player.on_ground:
            self._ground_move(vel, wish_dir, speed, dt)
        else:
            self._air_move(vel, wish_dir, speed, dt)

        self.player.velocity = vel

    # --------------------------------------------------
    # movement modes
    # --------------------------------------------------

    def _ground_move(self, vel, wish_dir, speed, dt):
        if np.linalg.norm(wish_dir) == 0.0:
            vel[0] = self._approach(vel[0], 0.0, self.deceleration * dt)
            vel[2] = self._approach(vel[2], 0.0, self.deceleration * dt)
            self.player.state = PlayerState.IDLE
            return

        target = wish_dir * speed
        vel[0] = self._approach(vel[0], target[0], self.acceleration * dt)
        vel[2] = self._approach(vel[2], target[2], self.acceleration * dt)

        self.player.state = PlayerState.SPRINT if speed > self.walk_speed else PlayerState.WALK

    def _air_move(self, vel, wish_dir, speed, dt):
        if np.linalg.norm(wish_dir) > 0.0:
            target = wish_dir * speed
            vel[0] = self._approach(
                vel[0], target[0], self.acceleration * self.air_control * dt
            )
            vel[2] = self._approach(
                vel[2], target[2], self.acceleration * self.air_control * dt
            )

        self.player.state = PlayerState.AIR

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    @staticmethod
    def _approach(current, target, max_delta):
        if current < target:
            return min(current + max_delta, target)
        if current > target:
            return max(current - max_delta, target)
        return target