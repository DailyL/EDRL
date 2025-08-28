from collision_risk.harm_estimation import cal_harm

from collision_risk.collision_probability import check_collisions, get_inv_mahalanobis_dist, get_collision_probability
from collision_risk.configs.load_json import load_harm_parameter_json, load_risk_json, load_weight_json
from collision_risk.risk_cost import cal_risk, get_bayesian_costs, get_equality_costs, get_maximin_costs, get_ego_costs
from collision_risk.helpers.prediction_helper import unprot_partic_trajs
from env.util.logger import risk_logger

paras_weights_ego = load_weight_json(filename="weights_ego.json")
paras_weights__ethical = load_weight_json(filename="weights_ethical.json")
paras_weights__standard = load_weight_json(filename="weights_standard.json")

modes = load_risk_json
harm = load_harm_parameter_json()


def calc_traj_costs(
        ego_traj,
        obj_predictions,
        engine,
        n_points,
        cost_fn_mode="standard",
        risk_harm_logger = None,
        only_vehicle = True,
        test_mode = False,
        
):
    """
    Main function to calculate the RL trajectory costs
    
    """
    
    if cost_fn_mode == "ego":
        weights = paras_weights_ego
    elif cost_fn_mode == "ethical":
        weights = paras_weights__ethical
    elif cost_fn_mode == "standard":
        weights = paras_weights__standard
    else:
        raise ValueError(f"Unknown cost function mode: {cost_fn_mode}")
    

    if not only_vehicle:
        # Consider add cyclist and perdestrain trajs 
        unprotected_road_users_trajs = unprot_partic_trajs(engine=engine, n_points=n_points)

        # Add perdestrain and cyclist for risk calculation
        obj_predictions = obj_predictions + unprotected_road_users_trajs

    ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max = cal_risk(
        ego_traj = ego_traj,
        obj_traj = obj_predictions,
    )

    if test_mode:

        new_obj_id_type = {}
        
        if obj_predictions is not None:
            for obj in obj_predictions:
                obj_id = obj['id']
                obj_type = obj['type']
                new_obj_id_type[obj_id] = obj_type



    if ego_risk_max is not None:

        if weights["risk_cost"] > 0.0 and obj_predictions is not None:

            bayes_cost = get_bayesian_costs(ego_risk_max, obj_risk_max)

            equality_cost = get_equality_costs(ego_risk_max, obj_risk_max)

            maximin_cost = get_maximin_costs(ego_risk_max, 
                                            obj_risk_max, 
                                            ego_harm_max, 
                                            obj_harm_max)
            
            ego_cost = get_ego_costs(ego_risk_max)

            # calculate risk cost
            total_risk_cost = (
                weights["bayes"] * bayes_cost
                + weights["equality"] * equality_cost
                + weights["maximin"] * maximin_cost
                + weights["ego"] * ego_cost
            )
            """ print("total risk with  ego", paras_weights_ego["bayes"] * bayes_cost
                + paras_weights_ego["equality"] * equality_cost
                + paras_weights_ego["maximin"] * maximin_cost
                + paras_weights_ego["ego"] * ego_cost)
            
            print("total risk with  ethical", paras_weights__ethical["bayes"] * bayes_cost
                + paras_weights__ethical["equality"] * equality_cost
                + paras_weights__ethical["maximin"] * maximin_cost
                + paras_weights__ethical["ego"] * ego_cost) """

            if test_mode:
                log_data = {
                    "ego_risk_max": ego_risk_max,
                    "obj_risk_max": obj_risk_max,
                    "ego_harm_max": ego_harm_max,
                    "obj_harm_max": obj_harm_max,
                    "bayes_cost": bayes_cost,
                    "equality_cost": equality_cost,
                    "maximin_cost": maximin_cost,
                    "ego_cost": ego_cost,
                    "obj_id_type": new_obj_id_type,
                }
                risk_harm_logger.log(log_data)
            

            return total_risk_cost
        else:
            return 1e-6
    else:
        return 1e-6

