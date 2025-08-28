from collision_risk.harm_estimation import cal_harm

from collision_risk.collision_probability import check_collisions, get_inv_mahalanobis_dist, get_collision_probability
from collision_risk.configs.load_json import load_harm_parameter_json, load_risk_json


params_mode = load_risk_json()
params_harm = load_harm_parameter_json()


def cal_risk(
    ego_traj,
    obj_traj,
):

    """
    Calculate the risk for the given trajectory. 
    """


    modes = params_mode
    coeffs = params_harm

    if modes["fast_prob_mahalanobis"]:
            coll_prob_dict = get_inv_mahalanobis_dist(
                traj = ego_traj,
                obj_predictions= obj_traj,
            )

    else:
        coll_prob_dict = get_collision_probability(
            traj = ego_traj,
            obj_prediction= obj_traj,
        )
    
    ego_harm_traj, obj_harm_traj = cal_harm(
         ego_traj = ego_traj,
         obj_prediction = obj_traj,
         modes = modes,
         coeffs = coeffs,
    )

    if ego_harm_traj is None or obj_harm_traj is None:
        return None, None, None, None



    # Calculate risk out of harm and collision probability
    ego_risk_traj = {}
    obj_risk_traj = {}
    ego_risk_max = {}
    obj_risk_max = {}
    ego_harm_max = {}
    obj_harm_max = {}

    for key in ego_harm_traj:
         ego_risk_traj[key] = [
            ego_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(ego_harm_traj[key]))
         ]

         obj_risk_traj[key] = [
            obj_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(obj_harm_traj[key]))
        ]
         
         # Take max as representative for the whole trajectory
         ego_risk_max[key] = max(ego_risk_traj[key])
         obj_risk_max[key] = max(obj_risk_traj[key])
         ego_harm_max[key] = max(ego_harm_traj[key])
         obj_harm_max[key] = max(obj_harm_traj[key])

    # Return the maximum risk and harm of the trajectries for each object
    # between (and) the ego vehicle
    return ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max




def get_bayesian_costs(ego_risk_max, obj_risk_max, boundary_harm = 0):
    """
    Bayesian Principle: The risk of all detected road users are cumulated and normalized.
    Minimizing the overall Risk

    Calculate the risk cost via the Bayesian Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obj_risk_max (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        Dict: Risk costs for the considered trajectory according to the
            Bayesian Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return (sum(ego_risk_max.values()) + sum(obj_risk_max.values()) + boundary_harm) / (
        len(ego_risk_max) * 2
    )


def get_equality_costs(ego_risk_max, obj_risk_max):
    """
    Equality Principle: distribute risks equally to all road users

    Calculate the risk cost via the Equality Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obj_risk_max (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        float: Risk costs for the considered trajectory according to the
            Equality Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(
        [abs(ego_risk_max[key] - obj_risk_max[key]) for key in ego_risk_max]
    ) / len(ego_risk_max)



def get_maximin_costs(ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max, boundary_harm = 0, eps=10e-10, scale_factor=100):
    """
    Maximin Principle: Minimizing the greatest possible harm regardless of probablity

    Calculate the risk cost via the Maximin principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        maximin_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Maximum Principle
    """
    if len(ego_harm_max) == 0:
        return 0

    # Set maximin to 0 if probability (or risk) is 0
    maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
    maximin_obj = [a * int(bool(b < eps)) for a, b in zip(obj_harm_max.values(), obj_risk_max.values())]

    return max(maximin_ego + maximin_obj + [boundary_harm])**scale_factor


def get_ego_costs(ego_risk_max, boundary_harm = 0):
    """
    Calculate the utilitarian ego cost for the given trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.

    Returns:
        Dict: Utilitarian ego risk cost
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(ego_risk_max.values()) + boundary_harm

