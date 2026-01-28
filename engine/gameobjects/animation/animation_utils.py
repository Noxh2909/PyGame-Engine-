import numpy as np
import math

# ------------------------------------------------------------
# Basic interpolation
# ------------------------------------------------------------

def lerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between vectors.
    """
    return v0 * (1.0 - t) + v1 * t


# ------------------------------------------------------------
# Quaternion helpers (glTF order: x, y, z, w)
# ------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion.
    """
    n = np.linalg.norm(q)
    if n == 0.0:
        return q
    return q / n


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Fallback to lerp for very small angles
        return quat_normalize(lerp(q0, q1, t))

    theta_0 = math.acos(dot)
    theta = theta_0 * t

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


# ------------------------------------------------------------
# Matrix construction
# ------------------------------------------------------------

def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def mat4_translate(v: np.ndarray) -> np.ndarray:
    """
    Build translation matrix from vec3.
    """
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v[0:3]
    return m


def mat4_scale(v: np.ndarray) -> np.ndarray:
    """
    Build scale matrix from vec3.
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = v[0]
    m[1, 1] = v[1]
    m[2, 2] = v[2]
    return m


def quat_to_mat4(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 4x4 rotation matrix.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m = np.eye(4, dtype=np.float32)

    m[0, 0] = 1.0 - 2.0 * (yy + zz)
    m[0, 1] = 2.0 * (xy - wz)
    m[0, 2] = 2.0 * (xz + wy)

    m[1, 0] = 2.0 * (xy + wz)
    m[1, 1] = 1.0 - 2.0 * (xx + zz)
    m[1, 2] = 2.0 * (yz - wx)

    m[2, 0] = 2.0 * (xz - wy)
    m[2, 1] = 2.0 * (yz + wx)
    m[2, 2] = 1.0 - 2.0 * (xx + yy)

    return m


# ------------------------------------------------------------
# TRS → local bone matrix
# ------------------------------------------------------------

def build_local_matrix(
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Build local bone matrix from TRS (glTF order).
    M_local = T * R * S
    """
    return (
        mat4_translate(translation)
        @ quat_to_mat4(rotation)
        @ mat4_scale(scale)
    )