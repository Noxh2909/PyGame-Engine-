import numpy as np

EDGE_EPS = 0.1  # Ground edge tolerance (0.05–0.15 works well)


class PhysicsWorld:
    """
    Minimal physics world for a kinematic player and static objects.
    """

    def __init__(self, gravity=(0.0, -25.0, 0.0)):
        self.gravity = np.array(gravity, dtype=np.float32)
        self.static_objects = []

    # -------------------------------------------------
    # World setup
    # -------------------------------------------------
    def add_static(self, obj):
        """
        Register a static (solid) object.
        The object MUST have:
        - obj.transform.position (vec3)
        - obj.collider with get_bounds(transform)
        """
        if obj.collider is None:
            raise ValueError("Static object must have a collider")
        self.static_objects.append(obj)

    # -------------------------------------------------
    # Physics step
    # -------------------------------------------------
    def step(self, dt, player):
        """
        Advance physics by dt seconds.
        Player is treated as a kinematic body.
        """
        # Apply gravity
        if not player.on_ground:
            player.velocity_y += self.gravity[1] * dt

        # Integrate vertical movement
        player.position[1] += player.velocity_y * dt

        # Resolve collisions
        self._resolve_player_collisions(player)

    # -------------------------------------------------
    # Collision resolution
    # -------------------------------------------------
    def _resolve_player_collisions(self, player):
        """
        Resolve collisions between the player (capsule-like) and static AABBs.
        Physics is the ONLY authority for grounding and blocking.
        """
        grounded = False

        for obj in self.static_objects:
            min_v, max_v = obj.collider.get_bounds(obj.transform)

            # -------------------------
            # X axis resolution
            # -------------------------
            if (
                min_v[1] <= player.position[1] <= max_v[1] + player.height
                and min_v[2] - player.radius
                <= player.position[2]
                <= max_v[2] + player.radius
            ):
                if (
                    player.position[0] > max_v[0]
                    and player.position[0] - player.radius < max_v[0]
                ):
                    player.position[0] = max_v[0] + player.radius
                elif (
                    player.position[0] < min_v[0]
                    and player.position[0] + player.radius > min_v[0]
                ):
                    player.position[0] = min_v[0] - player.radius

            # -------------------------
            # Z axis resolution
            # -------------------------
            if (
                min_v[1] <= player.position[1] <= max_v[1] + player.stand_height
                and min_v[0] - player.radius
                <= player.position[0]
                <= max_v[0] + player.radius
            ):
                if (
                    player.position[2] > max_v[2]
                    and player.position[2] - player.radius < max_v[2]
                ):
                    player.position[2] = max_v[2] + player.radius
                elif (
                    player.position[2] < min_v[2]
                    and player.position[2] + player.radius > min_v[2]
                ):
                    player.position[2] = min_v[2] - player.radius

            # -------------------------
            # Ground (Y axis) — ONLY when falling from above
            # -------------------------
            # Use standing height for grounding (posture-independent)
            ground_y = max_v[1] + player.stand_height

            was_above = player.prev_position[1] >= ground_y - 0.05
            is_crossing_down = player.position[1] <= ground_y

            if (
                was_above
                and is_crossing_down
                and player.velocity_y <= 0.0
                and min_v[0] - player.radius + EDGE_EPS
                <= player.position[0]
                <= max_v[0] + player.radius - EDGE_EPS
                and min_v[2] - player.radius + EDGE_EPS
                <= player.position[2]
                <= max_v[2] + player.radius - EDGE_EPS
            ):
                player.position[1] = ground_y
                player.velocity_y = 0.0
                grounded = True

        player.on_ground = grounded

        if grounded:
            player._jump_locked = False

    # -------------------------------------------------
    # Raycast (for camera collision)
    # -------------------------------------------------
    def raycast(self, start: np.ndarray, end: np.ndarray):
        """
        Raycast / segment test against static AABB colliders.

        Used for CAMERA collision only.
        Returns the closest hit point (np.ndarray) or None.
        """

        direction = end - start
        closest_t = 1.0
        hit_point = None

        for obj in self.static_objects:
            if obj.collider is None:
                continue

            min_v, max_v = obj.collider.get_bounds(obj.transform)

            tmin = 0.0
            tmax = 1.0

            # Slab method for segment vs AABB
            for axis in range(3):
                d = direction[axis]
                if abs(d) < 1e-6:
                    # Ray parallel to slab
                    if start[axis] < min_v[axis] or start[axis] > max_v[axis]:
                        break
                else:
                    inv_d = 1.0 / d
                    t1 = (min_v[axis] - start[axis]) * inv_d
                    t2 = (max_v[axis] - start[axis]) * inv_d

                    if t1 > t2:
                        t1, t2 = t2, t1

                    tmin = max(tmin, t1)
                    tmax = min(tmax, t2)

                    if tmin > tmax:
                        break
            else:
                # Valid intersection on the segment
                if tmin < closest_t:
                    closest_t = tmin
                    hit_point = start + direction * tmin

        return hit_point
