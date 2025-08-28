from generate_traj import preprocess_traj










if __name__ == "__main__":

    #env = make_wale_env(idm=True)
    #env = make_wale_real_env(idm=True)

    data_args = {
    "past_points": 30,  # number of past points as network input
    "future_points": 40,  # number of future points as ground truth for prediction
    "dpi": 300,  # dpi of rendered image
    "resolution": 256,  # resolution of the rendered image
    "watch_radius": 64,  # radius in m covered by scene image
    "exclude_handcrafted": True,  # exlcude handcrafted scenes
    "sliwi_size": 1,  # size of sliding window in dataset generation
    "shrink_percentage": 0.5,  # percentage of artificially added shrinked (with past points < "past_points") samples
    }

    

    trajs_path = "/home/dianzhaoli/EDRL/collision_risk/data/raw_data/13464/"


    preprocess_traj(
        args= data_args,
        trajs_path=trajs_path

    )