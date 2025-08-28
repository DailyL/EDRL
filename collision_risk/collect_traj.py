import os
import pickle
import numpy as np
import numpy as np
import pandas as pd
from scipy.io import savemat

collection_args = {
    "render": "self-rendered",  # Render Method: mpl or self-rendered
    "past_points": 30,  # number of past points as network input
    "future_points": 40,  # number of future points as ground truth for prediction
    "dpi": 300,  # dpi of rendered image
    "light_lane_div": True,  # show lane divider in image
    "resolution": 256,  # resolution of the rendered image
    "watch_radius": 64,  # radius in m covered by scene image
    "exclude_handcrafted": True,  # exlcude handcrafted scenes
    "sliwi_size": 1,  # size of sliding window in dataset generation
    "shrink_percentage": 0.5,  # percentage of artificially added shrinked (with past points < "past_points") samples
}

mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
data_directory = "data"
sc_img_dir = os.path.join(
        data_directory, "sc_imgs_small_{}/".format(str(collection_args["watch_radius"]))
    )
if not os.path.exists(sc_img_dir):
    os.makedirs(sc_img_dir)


pp = collection_args["past_points"]  # past points
fp = collection_args["future_points"]  # future points


file_path = "data/small.txt"


def vis_data(file_path):
    if isinstance(file_path, list):
        D = {"id": [], "hist": [], "fut": [], "nbrs": []}
        for file_p in file_path:
            with open(file_p, "rb") as fp:
                d = pickle.load(fp)
                for key in d:
                    D[key].extend(d[key])
    else:
        with open(file_path, "rb") as fp:
            D = pickle.load(fp)

    print((D["nbrs"]))


def parse_scene(env):
    
    obs = env.reset()
    assert isinstance(
        obs,
        (np.ndarray, dict),
    ), "Tuple observations are not supported."
    
    active = np.ones(env.num_envs, dtype=bool)
    state = None
    dones = np.zeros(env.num_envs, dtype=bool)

    while np.any(active):
        # policy gets unwrapped observations (eg as dict, not dictobs)
        obs, rews, dones, infos = env.step([0,0])
        acts = (np.array(infos[0]['raw_action'])).reshape(1, -1)
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active
    
    
    


hist_list = []
fut_list = []
nbrs_list = []
id_list = []




    
    
    









# Input files
us101_1 = '/home/dianzhaoli/scenario_dataset/NGSIM/us101/trajectories-0750am-0805am.txt'
us101_2 = '/home/dianzhaoli/scenario_dataset/NGSIM/us101/trajectories-0805am-0820am.txt'
us101_3 = '/home/dianzhaoli/scenario_dataset/NGSIM/us101/trajectories-0820am-0835am.txt'
i80_1 = '/home/dianzhaoli/scenario_dataset/NGSIM/i80/trajectories-0400-0415.txt'
i80_2 = '/home/dianzhaoli/scenario_dataset/NGSIM/i80/trajectories-0500-0515.txt'
i80_3 = '/home/dianzhaoli/scenario_dataset/NGSIM/i80/trajectories-0515-0530.txt'

# Load data
def load_data(file, dataset_id):
    data = np.loadtxt(file)
    # Keep relevant columns and initialize additional columns (6 extra for maneuvers and 39 for grid locations)
    new_data = np.zeros((data.shape[0], 47), dtype=np.float32)
    new_data[:, :6] = data[:, [0, 1, 2, 5, 6, 14]]  # Copy columns 0, 1, 2, 5, 6, 14 from original data
    new_data[:, 0] = dataset_id  # Set the dataset ID
    return new_data

traj = [
    load_data(us101_1, 1),
    load_data(us101_2, 2),
    load_data(us101_3, 3),
    load_data(i80_1, 4),
    load_data(i80_2, 5),
    load_data(i80_3, 6)
]

# Adjust data, limit lanes to 6 in first 3 datasets
for k in range(6):
    traj[k] = traj[k][:, [0, 1, 2, 5, 6, 14]]
    if k <= 2:
        traj[k][traj[k][:, 5] >= 6, 5] = 6

# Vehicle trajectories and frame times
vehTrajs = [{} for _ in range(6)]
vehTimes = [{} for _ in range(6)]

# Parse fields
for ii in range(5):
    vehIds = np.unique(traj[ii][:, 1])
    
    for v in vehIds:
        vehTrajs[ii][str(int(v))] = traj[ii][traj[ii][:, 1] == v, :]
    
    timeFrames = np.unique(traj[ii][:, 2])
    
    for v in timeFrames:
        vehTimes[ii][str(int(v))] = traj[ii][traj[ii][:, 2] == v, :]
    
    for k in range(len(traj[ii])):
        time = traj[ii][k, 2]
        dsId = traj[ii][k, 0]
        vehId = traj[ii][k, 1]
        vehtraj = vehTrajs[ii][str(int(vehId))]
        ind = np.where(vehtraj[:, 2] == time)[0][0]
        lane = traj[ii][k, 5]
        
        # Lateral maneuver
        ub = min(len(vehtraj) - 1, ind + 40)
        lb = max(0, ind - 40)
        # Check lateral maneuver
        if len(vehtraj) > 1:  # Ensure vehtraj has enough rows
            if vehtraj[ub, 5] > vehtraj[ind, 5] or vehtraj[ind, 5] > vehtraj[lb, 5]:
                traj[ii][k, 6] = 3
            elif vehtraj[ub, 5] < vehtraj[ind, 5] or vehtraj[ind, 5] < vehtraj[lb, 5]:
                traj[ii][k, 6] = 2
            else:
                traj[ii][k, 6] = 1
        else:
            traj[ii][k, 6] = 1  # Default value if vehtraj doesn't have enough data
        
        # Longitudinal maneuver
        ub = min(len(vehtraj), ind + 50)
        lb = max(0, ind - 30)
        if ub == ind or lb == ind:
            traj[ii][k, 7] = 1
        else:
            vHist = (vehtraj[ind, 4] - vehtraj[lb, 4]) / (ind - lb)
            vFut = (vehtraj[ub, 4] - vehtraj[ind, 4]) / (ub - ind)
            if vFut / vHist < 0.8:
                traj[ii][k, 7] = 2
            else:
                traj[ii][k, 7] = 1

        # Grid locations
        t = vehTimes[ii][str(int(time))]
        frameEgo = t[t[:, 5] == lane]
        frameL = t[t[:, 5] == lane - 1]
        frameR = t[t[:, 5] == lane + 1]
        
        for frame, shift in zip([frameL, frameEgo, frameR], [1, 14, 27]):
            if len(frame) > 0:
                for l in range(len(frame)):
                    y = frame[l, 4] - traj[ii][k, 4]
                    if abs(y) < 90 and (shift != 14 or y != 0):
                        gridInd = shift + round((y + 90) / 15)
                        traj[ii][k, 8 + gridInd] = frame[l, 1]

# Split into training, validation, and test sets
traj_all = np.vstack(traj)
traj_tr = []
traj_val = []
traj_ts = []

for k in range(1, 7):
    ul1 = round(0.7 * max(traj_all[traj_all[:, 0] == k, 1]))
    ul2 = round(0.8 * max(traj_all[traj_all[:, 0] == k, 1]))
    
    traj_tr.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] <= ul1), :])
    traj_val.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] > ul1) & (traj_all[:, 1] <= ul2), :])
    traj_ts.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] > ul2), :])

traj_tr = np.vstack(traj_tr)
traj_val = np.vstack(traj_val)
traj_ts = np.vstack(traj_ts)

# Create track dictionaries for train, val, test sets
def create_tracks(traj):
    tracks = {}
    for k in range(1, 7):
        traj_set = traj[traj[:, 0] == k, :]
        car_ids = np.unique(traj_set[:, 1])
        tracks[k] = {int(car_id): traj_set[traj_set[:, 1] == car_id, 2:5].T for car_id in car_ids}
    return tracks

tracks_tr = create_tracks(traj_tr)
tracks_val = create_tracks(traj_val)
tracks_ts = create_tracks(traj_ts)

# Filter edge cases (skip initial 3 seconds)
def filter_edge_cases(traj, tracks):
    inds = np.zeros(len(traj), dtype=bool)
    for k in range(len(traj)):
        t = traj[k, 2]
        track = tracks[int(traj[k, 0])][int(traj[k, 1])]
        if track[0, 30] <= t and track[0, -1] > t + 1:
            inds[k] = True
    return traj[inds]

traj_tr = filter_edge_cases(traj_tr, tracks_tr)
traj_val = filter_edge_cases(traj_val, tracks_val)
traj_ts = filter_edge_cases(traj_ts, tracks_ts)

# Save as mat files
savemat('TrainSet.mat', {'traj': traj_tr, 'tracks': tracks_tr})
savemat('ValSet.mat', {'traj': traj_val, 'tracks': tracks_val})
savemat('TestSet.mat', {'traj': traj_ts, 'tracks': tracks_ts})

print("Data saved to TrainSet.mat, ValSet.mat, and TestSet.mat.")
