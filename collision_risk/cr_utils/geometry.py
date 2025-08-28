import numpy as np
import sys

def transform_trajectories(trajectories_list, now_point, theta):
    """Transform a list of trajectories by translation and rotation.

    Arguments:
        trajectories_list {[list]} -- [list of trajectories being transformed]
        now_point {[list]} -- [[x,y] translation]
        orientation {[float]} -- [rotation]
    Returns:

        [trans_traj_list] -- [tranformed trajectory list]
    """
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    trans_traj_list = []

    # Check if it's a 1D trajectory list or 2D
    for trajectories in trajectories_list:
        for tr in trajectories:
            tr_1 = np.array(tr) - np.array(now_point)  # Translate
            tr_2 = np.matmul(tr_1, rot_mat)  # Rotate
            trans_traj_list.append(tr_2)

    return trans_traj_list

def transform_back(trajectory, translation, rotation):
    """Back transformation of a single trajectory

    Arguments:
        trajectory {[list]} -- [Trajectory points in x,y]
        translation {[list]} -- [[x,y] translation]
        rotation {[float]} -- [rotation]

    Returns:
        [type] -- [description]
    """
    rotation = -rotation
    translation = -translation
    rot_mat = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    trajectory[:, :2] = np.matmul(trajectory[:, :2], rot_mat)
    trajectory[:, :2] = trajectory[:, :2] - translation

    # Transform sigmas
    if trajectory.shape[1] > 2:
        sigma_x = 1 / (trajectory[:, 2] + sys.float_info.epsilon)
        sigma_y = 1 / (trajectory[:, 3] + sys.float_info.epsilon)
        rho = trajectory[:, 4]

        sigma_cov = np.array(
            [
                [sigma_x ** 2, rho * sigma_x * sigma_y],
                [rho * sigma_x * sigma_y, sigma_y ** 2],
            ]
        )

        # Swap axes to shape (50,2,2)
        sigma_cov = sigma_cov.swapaxes(0, 2)
        sigma_cov = sigma_cov.swapaxes(1, 2)

        # sigma_cov_trans = np.matmul(rot_mat.T, sigma_cov)
        # sigma_cov_trans = np.matmul(sigma_conv_trans, rot_mat)
        # print(np.linalg.eig(sigma_cov_trans[0,:,:]))

        sigma_cov_trans = rot_mat.T @ sigma_cov @ rot_mat

        return trajectory[:, :2], sigma_cov_trans

    else:
        return trajectory[:, :2]
    

def point_in_rectangle(r1, r2, p):
    """Calculates weather a point p is in a recangle defined by two corner points r1 and r2.
    If true the position in relative coordinates between 0 and 2 in and 0 and 12 in y are returned.

    Arguments:
        r1 {[list]} -- [[x,y] point of rectancle corner]
        r2 {[list]} -- [[x,y] point of rectancle corner]
        p {[list]} -- [[x,y] point which is tested to be in that rectanle]

    Returns:
        [bool / tuple] -- [False if point is not in rectangle, else relative position]
    """
    bottom_left = [min(r1[0], r2[0]), min(r1[1], r2[1])]
    top_right = [max(r1[0], r2[0]), max(r1[1], r2[1])]

    if (
        p[0] > bottom_left[0]
        and p[0] < top_right[0]
        and p[1] > bottom_left[1]
        and p[1] < top_right[1]
    ):
        x = int((p[0] - bottom_left[0]) / (top_right[0] - bottom_left[0]) * 3)
        y = int((top_right[1] - p[1]) / (top_right[1] - bottom_left[1]) * 13)
        return (x, y)
    else:
        return False