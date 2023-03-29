import numpy as np

def transform_normals(X_AB, p_B) -> np.ndarray:
  p_A = p_B @ X_AB[:3,:3].T
  return p_A

def transform_points(X_AB, p_B) -> np.ndarray:
  """
  Args:
    X_AB: 4x4 matrix
    p_B: (N, 3) 
  Returns:
    p_A: (N, 3)
  """
  p_A = homogenize_points(p_B) @ X_AB.T 
  p_A = unhomogenize_points(p_A)
  return p_A

def homogenize_points(p_A) -> np.ndarray:
  """
  Args:
    p_A: (N,3)
  Returns:
    p_A: (N,4)
  """
  n_points = p_A.shape[0]
  return np.concatenate([p_A, np.ones((n_points, 1), dtype=np.float32)], axis=-1)

def unhomogenize_points(p_A) -> np.ndarray:
  """
  Args:
    p_A: (N,4)
  Returns:
    p_A: (N,3)
  """
  p_A[..., :3] /= p_A[..., -1:] 
  return p_A[..., :3]

def normalize(a: np.ndarray) -> np.ndarray:
  return a / np.linalg.norm(a)

