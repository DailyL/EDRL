import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2



def prob_with_mahalanobis_chi_squared(ego_pose, obj_pose, cov_mat, collision_threshold = 0.0):
    """
    Calculate the collision probability using the Mahalanobis distance
    and the chi-squared distribution.

    Parameters:
    r (np.array): The relative position vector (x, y) between two vehicles.
    cov (np.array): The combined covariance matrix (2x2).
    collision_threshold (float): The threshold for collision detection (e.g., size of bounding boxes).

    Returns:
    float: Collision probability.
    """
    # Compute the Mahalanobis distance
    d_m = mahalanobis(ego_pose, obj_pose, cov_mat)

    
    prob = 1-chi2.cdf(d_m, df=2)

    # Compute the collision probability using a chi-squared distribution
    return prob


ego_pose = np.array([0, 0])
obj_pose = np.array([0.9, 0.1])
cov_mat = np.array([[1, 0], [0, 1]])

print(prob_with_mahalanobis_chi_squared(ego_pose, obj_pose, cov_mat))