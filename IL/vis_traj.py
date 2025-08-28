from generate_demos import load_traj

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import generate_gif
from imitation.data.rollout import rollout_stats


def simple_env(acts):
    env=MetaDriveEnv(dict(map="S", traffic_density=0))
    frames = []
    try:
        env.reset()
        cfg=env.config["vehicle_config"]
        cfg["navigation"]=None # it doesn't need navigation system
        v = env.engine.spawn_object(DefaultVehicle, 
                                    vehicle_config=cfg, 
                                    position=[30,0], 
                                    heading=0)
        for i in range(len(acts)):
            v.before_step([0, 0.5])
            env.step(acts[i,:])
            frame=env.render(mode="topdown", 
                            window=False,
                            screen_size=(800, 200),
                            camera_position=(60, 7))
            frames.append(frame)
        generate_gif(frames, gif_name="demo.gif")
    finally:
        env.close()





def print_acts(acts):
    a1 = acts[0]
    a2 = acts[1]
    max_steer = 40
    max_engine_force = 750
    max_brake_force = 120
    final = []
    steers = []
    accs = []
    braks = []
    for row in acts:
        steer = max_steer * row[0]
        acc = max_engine_force * max(0, row[1])
        brak = -max_brake_force * min(0, row[1])
        steers.append(steer)
        accs.append(acc)
        braks.append(brak)





if __name__ == "__main__":

    expert_name = "Random"
    demo_name = "SUMO"
    num_episode = 1
    trainer = 'gail'

    
    transitions = load_traj(
        expert_name=expert_name, 
        scenario_name=demo_name,
        trainer_name=trainer,
        num_scenarios=num_episode,
        )
    #print((transitions[0].acts))
    simple_env((transitions[0].acts))
