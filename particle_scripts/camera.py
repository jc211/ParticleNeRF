import numpy as np
from typing import NamedTuple 
from transforms import *

class Camera(NamedTuple):
  X_WV: np.ndarray
  K: np.ndarray
  image_width: int
  image_height:int

  def copy_with_new_pose(self, X_WV):
      X = np.eye(4)
      X[:3, :4] = X_WV[:3, :4]
      return self._replace(X_WV=X)
    

  def get_3x3_projection_matrix(self) -> np.ndarray:
      K = np.eye(3)
      K[0, 0] = self.K[0, 0]
      K[1, 1] = self.K[1, 1]
      K[0, 2] = self.K[0, 2]
      K[1, 2] = self.K[1, 2]
      return K

  def world_to_view_matrix(self) -> np.ndarray:
    """
    Inverse of X_VW
    """
    X_VW = np.linalg.inv(self.X_WV)
    return X_VW

  def get_camera_center(self) -> np.ndarray:
    return self.X_WV[:3, 3]

  def unproject_points(self, points, world_coordinates:bool = True):
    """
    Args:
      points: np.ndarray (N, 3) (x_screen, y_screen, depth)
      world_coordinates: returns points in world, otherwise in view
    Returns
      p_W: np.ndarray (N,3) in world coordinates
    """
    # p (4, n_points)
    shape = points.shape
    points = points.reshape(-1, 3)

    p = np.stack([points.T[0], points.T[1], 1.0/points.T[2]], axis=0).T
    X_SV = self.K
    X_VS = np.linalg.inv(X_SV)
    p_V = transform_points(X_VS, p)
    if world_coordinates:
      p_W = transform_points(self.X_WV, p_V)
      return p_W.reshape(shape)
    else:
      return p_V.reshape(shape)

def create_projection_matrix(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0], 
    ], dtype=np.float32) # Get it to convert from blender view convention rather than opencv

def projection_matrix_3x3_to_4x4(K: np.ndarray) -> np.ndarray:
    """Convert
     K = [
            [fx,   0,    px],
            [0,   fy,    py],
            [0,    0,     1],
     ] to
      K = [
            [fx,   0,   px,   0],
            [0,   fy,   py,   0],
            [0,    0,    0,   1],
            [0,    0,    1,   0],
    ]
    """
    Kn = np.copy(K)
    Kn = np.insert(Kn, 2, 0, axis=0)
    Kn = np.insert(Kn, 3, 0, axis=1)
    Kn[2, 3] = 1
    return Kn