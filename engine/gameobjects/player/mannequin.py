import math
import numpy as np

def lerp_angle(a, b, t):
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t

class Mannequin:
    def __init__(self, player, mesh, material, foot_offset, scale):
        self.player = player
        self.mesh = mesh
        self.material = material
        self.foot_offset = foot_offset
        self.scale_factor = scale

        # visual-only yaw (radians), smoothed follow of player direction
        self.visual_yaw = 0.0
        
        # local yaw offset to give a bit of angle to the mannequin
        self.local_yaw_offset = math.radians(5.0)  # 10–25° sinnvoll

        # how fast the mannequin follows the player yaw (0.1 = soft, 0.3 = snappy)
        self.yaw_follow_strength = 0.25

    def matrix(self):
        pos = self.player.position.copy().astype(np.float32)

        # capsule center → ground
        pos[1] -= self.player.height * self.player.stand_height

        # ground → feet
        pos[1] += self.foot_offset * self.scale_factor

        # small epsilon to avoid z-fighting / sinking
        pos[1] += 0.05

        # yaw MUST be radians
        front = self.player.front
        target_yaw = math.atan2(front[0], front[2])

        # smooth visual rotation towards player direction
        self.visual_yaw = lerp_angle(
            self.visual_yaw,
            target_yaw,
            self.yaw_follow_strength
        )

        yaw = self.visual_yaw
        yaw += self.local_yaw_offset
        c, s = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [ c, 0,  s, 0],
            [ 0, 1,  0, 0],
            [-s, 0,  c, 0],
            [ 0, 0,  0, 1],
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = self.scale_factor

        return T @ R @ S