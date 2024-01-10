"""
    This file contains the FrameConverter class, which is used to convert
    poses between common coordinate frames.
"""
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation


class FrameConverter:
    """
    Works using rotations from: https://github.com/dvdmc/frame-transformer
    The unified representation are quaternions from scipy.spatial.transform.Rotation
    Conventions:
    - ROS uses x-front y-left z-up (From: https://github.com/ethz-asl/unreal_airsim/blob/master/src/frame_converter.cpp)
    - AirSim uses x-front y-right z-down
    - COLMAP uses x-right y-down z-forward (This corresponds to a rotation of 120 degrees around the axis [-1,-1,-1] in Airsim)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.rotation = Rotation.identity()
        self.rotation_inv = Rotation.identity()

    def setup_from_yaw(self, yaw: float):
        yaw_offset = yaw % (2 * np.pi)
        rotation = Rotation.from_rotvec([0, 0, yaw_offset])
        self.rotation = rotation
        self.rotation_inv = rotation.inv()
    

    def setup_transform_function(self, from_frame: str, to_frame: str) -> None:
        """Gets the transform function from one frame to another
        Args:
            from_frame: frame to transform from
            to_frame: frame to transform to
        Returns:
            transform function
        """
        if from_frame == "ros" and to_frame == "airsim":
            self.transform_function = self.ros_to_airsim_pose
        elif from_frame == "airsim" and to_frame == "ros":
            self.transform_function = self.airsim_to_ros_pose
        elif from_frame == "airsim" and to_frame == "colmap":
            self.transform_function = self.airsim_to_colmap_pose
        elif from_frame == "colmap" and to_frame == "airsim":
            self.transform_function = self.colmap_to_airsim_pose
        else:
            raise NotImplementedError
        

    def ros_to_airsim_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts world pose to AirSim pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), quaternion (4x1)
        """
        # Transform translation
        p = np.array([t[0], t[1], t[2]])
        # Apply rotation to point
        # Axis flip
        x = p[0]
        y = -p[1]
        z = -p[2]
        
        p = self.rotation_inv.apply(p)

        # Transform rotation
        rotation_ros_to_airsim = np.array([[1,0,0],
                                           [0,-1,0],
                                           [0,0,-1]])
        matrix_rot = rot.as_matrix()
        base_change = rotation_ros_to_airsim @ matrix_rot @ rotation_ros_to_airsim.T
        rotation =  self.rotation * Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation

    def airsim_to_ros_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts AirSim pose to world pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        p = np.array([t[0], -t[1], -t[2]])
        # Apply rotation to point
        p = self.rotation.apply(p)
        
        x = p[0]
        y = -p[1]
        z = -p[2]

        rotation_airsim_to_ros = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        yaw_rotated = self.rotation_inv * rot
        rot_matrix = yaw_rotated.as_matrix()
        base_change = rotation_airsim_to_ros @ rot_matrix @ rotation_airsim_to_ros.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation

    def airsim_to_colmap_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts AirSim pose to COLMAP pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Swap axes
        p = np.array([t[1], t[2], t[0]])
        # Apply rotation to point
        p = self.rotation.apply(p)
        x = p[0]
        y = p[1]
        z = p[2]

        # Compute rotation TODO: Do we apply rotation?
        rotation_airsim_to_colmap = np.array([[0,1,0],[0,0,1],[1,0,0]])
        rot_matrix = rot.as_matrix()
        base_change = rotation_airsim_to_colmap @ rot_matrix @ rotation_airsim_to_colmap.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation
    

    def colmap_to_airsim_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts COLMAP pose to AirSim pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        p = np.array([t[0], t[1], t[2]])
        # Apply rotation to point
        p = self.rotation_inv.apply(p)
        # Axis flip
        x = p[2]
        y = p[0]
        z = p[1]

        # Transform rotation TODO: Do we apply rotation?
        rotation_colmap_to_airsim = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        rot_matrix = rot.as_matrix()
        base_change = rotation_colmap_to_airsim @ rot_matrix @ rotation_colmap_to_airsim.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation

    def colmap_to_ros_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts COLMAP pose to world pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        p = np.array([t[0], t[1], t[2]])
        # Apply rotation to point
        # TODO: check if there is any instance when this applies
        #  p = self.rotation_inv * p
        # Axis flip
        x = p[2]
        y = -p[0]
        z = -p[1]

        # Transform rotation
        rotation_colmap_to_ros = np.array([[0, 0, 1], 
                                           [-1, 0, 0], 
                                           [0, -1, 0]])
        rot_matrix = rot.as_matrix()
        base_change = rotation_colmap_to_ros @ rot_matrix @ rotation_colmap_to_ros.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation

    def ros_to_colmap_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts world pose to COLMAP pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        p = np.array([t[0], t[1], t[2]])
        # Apply rotation to point
        # p = self.rotation_inv * p
        # Axis flip
        x = -p[1]
        y = -p[2]
        z = p[0]

        # Transform rotation
        rotation_ros_to_colmap = np.array([[0, -1, 0], 
                                            [0, 0, -1], 
                                            [1, 0, 0]])
        rot_matrix = rot.as_matrix()
        base_change = rotation_ros_to_colmap @ rot_matrix @ rotation_ros_to_colmap.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation
    
    def ros_to_nerfstudio_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts world pose to NerfStudio pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        # Axis flip
        x = t[1]
        y = t[2]
        z = t[0]

        # Transform rotation
        rotation_ros_to_nerfstudio = np.array([[0, 1, 0], 
                                               [0, 0, 1], 
                                               [1, 0, 0]])
        rot_matrix = rot.as_matrix()
        base_change = rotation_ros_to_nerfstudio @ rot_matrix @ rotation_ros_to_nerfstudio.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation


    def nerfstudio_to_ros_pose(
        self, t: np.ndarray, rot: Rotation
    ) -> Tuple[np.ndarray, Rotation]:
        """Converts NerfStudio pose to world pose
        Args:
            t: translation (3x1)
            rot: rotation from scipy
        Returns:
            translation (3x1), rotation from scipy
        """
        # Transform translation
        # Axis flip
        x = t[2]
        y = t[0]
        z = t[1]

        # Transform rotation
        rotation_nerfstudio_to_ros = np.array([[0, 0, 1], 
                                               [1, 0, 0], 
                                               [0, 1, 0]])
        
        rot_matrix = rot.as_matrix()
        base_change = rotation_nerfstudio_to_ros @ rot_matrix @ rotation_nerfstudio_to_ros.T
        rotation = Rotation.from_matrix(base_change)

        return np.array([x, y, z]), rotation
