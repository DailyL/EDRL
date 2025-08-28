import os, pickle, glob
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import generate_gif
from imitation.data.rollout import rollout_stats
from imitation.data import rollout, serialize


def exam_traj(save_paths):
    for filename in save_paths:
        if os.path.getsize(filename) > 0:
                try:
                    with open(filename, 'rb') as file:
                        data = pickle.load(file)
                except EOFError:
                    print(f"Error: The file {filename} is incomplete or corrupted.")
        else:
            print(f"The file '{filename}' is empty.")

def load_traj_multi(trainer_name):

    rollouts_path = Path(f"../IL_rollouts/{trainer_name}")
    pattern = "*.pkl"

    rollouts_names =  glob.glob(os.path.join(rollouts_path, pattern))

    transitions = exam_traj(rollouts_names)


if __name__ == "__main__":

    trainer = "gail"

    transitions = load_traj_multi(
                trainer_name=trainer,
                )