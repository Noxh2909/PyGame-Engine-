# import numpy as np


# class PlayerPhysics:
#     """
#     Handles physical simulation of the player:
#     - position integration
#     - collision resolution
#     - ground detection

#     Movement logic (walk/jump/etc.) lives in PlayerMovement.
#     """

#     def __init__(self, world):
#         """
#         world.objects must contain GameObjects with AABB colliders
#         """
#         self.world = world

#     def step(self, player, dt: float):
#         # --- integrate velocity ---
#         pos = player.transform.position
#         vel = player.velocity

#         old_pos = pos.copy()
#         new_pos = old_pos + vel * dt

#         player.on_ground = False

#         # --- resolve collisions ---
#         for obj in self.world.objects:
#             collider = obj.collider
#             if collider is None:
#                 continue

#             aabb_min, aabb_max = collider.get_aabb(obj.transform.position)

#             if not player.collider.intersects_aabb(new_pos, aabb_min, aabb_max):
#                 continue

#             # ---------- vertical resolution (ground / ceiling) ----------
#             if vel[1] <= 0.0:
#                 # landing on top
#                 ground_y = aabb_max[1]
#                 new_pos[1] = ground_y
#                 vel[1] = 0.0
#                 player.on_ground = True
#             else:
#                 # hit ceiling
#                 new_pos[1] = old_pos[1]
#                 vel[1] = 0.0

#             # ---------- horizontal resolution (slide) ----------
#             # resolve X
#             test = new_pos.copy()
#             test[0] = old_pos[0]
#             if not player.collider.intersects_aabb(test, aabb_min, aabb_max):
#                 new_pos[0] = old_pos[0]

#             # resolve Z
#             test = new_pos.copy()
#             test[2] = old_pos[2]
#             if not player.collider.intersects_aabb(test, aabb_min, aabb_max):
#                 new_pos[2] = old_pos[2]

#         # --- apply resolved position ---
#         player.transform.position[:] = new_pos
