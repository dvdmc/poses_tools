###
#
# This file includes transformation functions between different representations of poses.
#
###

from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> Rotation:
    """Converts roll, pitch, yaw to rotation matrix
    Args:
        roll: roll
        pitch: pitch
        yaw: yaw
    Returns:
        rotation object from scipy
    """
    return Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        

def rotation_to_rpy(rotation: Rotation) -> np.ndarray:
    """Converts rotation matrix to roll, pitch, yaw
    Args:
        rotation: rotation transform from scipy
    Returns:
        roll, pitch, yaw
    """
    return rotation.as_euler("xyz", degrees=False)


def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[np.ndarray, Rotation]:
    """Converts spherical coordinates to cartesian coordinates
        where the orientation points towards the center of the sphere.
    Args:
        r: radius
        theta: angle from the z axis to the x-y plane
        phi: angle in the x-y plane around the z axis
    Returns:
        translation x, y, z
        rotation from scipy
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    rotation = Rotation.from_euler("xyz", [0, -np.pi/2+theta, phi], degrees=False)

    translation = np.array([x, y, z])
    return translation, rotation


def transform_matrix_to_pose(matrix: np.ndarray) -> Tuple[np.ndarray, Rotation]:
    """Converts a 4x4 transform matrix to a pose
    Args:
        matrix: 4x4 transform matrix
    Returns:
        translation x, y, z
        rotation from scipy
    """
    translation = matrix[:3, 3]
    rotation = Rotation.from_matrix(matrix[:3, :3])
    return translation, rotation


def t_q_to_pose(t: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, Rotation]:
    """Converts translation and quaternion to pose
    Args:
        t: translation (3x1)
        q: quaternion (4x1)
    Returns:
        translation x, y, z
        rotation from scipy
    """
    rotation = Rotation.from_quat(q)
    return t, rotation

"""
    The following is only kept for the equations.
"""
def old_rpy_to_quaternion(r: float, p: float, y: float) -> np.ndarray:
    """Converts roll, pitch, yaw to quaternion
    Args:
        r: roll
        p: pitch
        y: yaw
    Returns:
        q: quaternion (w, x, y, z)
    """
    t0 = np.cos(y * 0.5)
    t1 = np.sin(y * 0.5)
    t2 = np.cos(r * 0.5)
    t3 = np.sin(r * 0.5)
    t4 = np.cos(p * 0.5)
    t5 = np.sin(p * 0.5)

    q = np.zeros(4)
    q[0] = t0 * t2 * t4 + t1 * t3 * t5
    q[1] = t0 * t3 * t4 - t1 * t2 * t5
    q[2] = t0 * t2 * t5 + t1 * t3 * t4
    q[3] = t1 * t2 * t4 - t0 * t3 * t5

    return q

def old_quaternion_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    """Converts quaternion to roll, pitch, yaw
    Args:
        q: quaternion (w, x, y, z)
    Returns:
        roll, pitch, yaw
    """
    z = q[3]
    y = q[2]
    x = q[1]
    w = q[0]
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1
    if t2 < -1.0:
        t2 = -1.0
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw