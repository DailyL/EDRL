import csv
import random
import shutil
import time
import argparse
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import pathlib

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data

import tud_rl.agents.continuous as agents
from tud_rl import logger
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.logging_plot import plot_from_progress

from tud_rl.wrappers import get_wrapper


# ENV imports

from env.make_env import make_tudrl_purestate, make_vec_env, make_vec_env_purestate, make_srl_purestate, make_real_env, make_real_no_cost_env

def evaluate_policy(test_env: gym.Env, agent: _Agent, c: ConfigFile):

    # go greedy
    agent.mode = "test"

    rets = []
    costs = []
    costs_per_step = []

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if agent.needs_history:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        if c.Env.name == "SRL_Traj_Env":
            s, info = test_env.reset()
        else:
            s = test_env.reset()

        cur_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
        cur_cost = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # select action
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # perform step
            if c.Env.name == "UAM-v0" and agent.name == "LSTMRecTD3":
                s2, r, d, _ = test_env.step(agent)
            else:
                s2, r, d, info = test_env.step(a)

            # LSTM: update history
            if agent.needs_history:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1

            # s becomes s2
            s = s2
            cur_ret += r
            cur_cost += info["cost"]

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break

        # append return
        rets.append(cur_ret)
        costs.append(cur_cost)
        costs_per_step.append(cur_cost / eval_epi_steps)

    # continue training
    agent.mode = "train"
    return rets, costs, costs_per_step

def visualize_policy(env: gym.Env, agent: _Agent, c: ConfigFile):

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if agent.needs_history:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        if c.Env.name == "SRL_Traj_Env":
            s, info = env.reset()
        else:
            s = env.reset()

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # render env
            #env.render()

            # select action
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # perform step
            if c.Env.name == "UAM-v0" and agent.name == "LSTMRecTD3":
                s2, r, d, _ = env.step(agent)
                
            
            elif c.Env.name == "HHOS-OpenPlanning-Validation-v0" and "full_RL" in c.Env.env_kwargs:
                if c.Env.env_kwargs["full_RL"]:
                    s2, r, d, _ = env.step([a, agent])
                else:
                    s2, r, d, _ = env.step(a)
            else:
                s2, r, d, _ = env.step(a)
            # LSTM: update history
            if agent.needs_history:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break
        print(cur_ret)

def load_scenarios(parent_path, dataset_lists):

    len_datasets = len(dataset_lists)
    seed = random.randint(0, len_datasets-1)
    dataset = dataset_lists[seed]

    path = pathlib.Path(AssetLoader.file_path(AssetLoader.asset_path, f'{parent_path}/scenario-nuplan-{dataset}', unix_style=False))
    summary, scenario_ids, mapping = read_dataset_summary(path)
    len_scenarios = len(scenario_ids)

    return len_scenarios, path, mapping

   


def train(config_path, env, agent_name: str):
    """Main training loop."""
    # parse the config file
    c = ConfigFile(f"{config_path}")
    seed = random.randint(0, 20000)
    # measure computation time
    start_time = time.time()

    # init envs
    env = env
    test_env = env

    # get state_shape
    c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "train"
    c.num_actions = 3

    # seeding
    env.seed(seed)
    test_env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # get agent class by name
    
    agent: _Agent = agent_(c, agent_name)  # instantiate agent
    
    if agent.name == "LSTMSACLAG" or agent.name == "SACLAG":
        agent.constraint_threshold = cost_limit

    # initialize logging
    agent.logger = EpochLogger(alg_str    = agent.name,
                               seed       = seed,
                               env_str    = c.Env.name,
                               info       = c.Env.info,
                               output_dir = c.output_dir if hasattr(c, "output_dir") else None)

    agent.logger.save_config({"agent_name": agent.name, **c.config_dict})
    agent.print_params(agent.n_params, case=1)


    # LSTM: init history
    if agent.needs_history:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

    # get initial state
    if c.Env.name == "SRL_Traj_Env":
        s, info = env.reset()
    else:
        s = env.reset()

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
    epi_cost = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0


    # main loop
    for total_steps in range(c.timesteps):
        epi_steps += 1

        # select action
        if total_steps < c.act_start_step:
            if agent.is_multi:
                a = np.random.uniform(low=-1.0, high=1.0, size=(agent.N_agents, agent.num_actions))
            else:
                a = np.random.uniform(low=-1.0, high=1.0, size=agent.num_actions)
        else:
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

        # perform step
        if c.Env.name == "UAM-v0" and agent.name == "LSTMRecTD3":
            s2, r, d, _ = env.step(agent)
        else:
            s2, r, d, info = env.step(a)


        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c.Env.max_episode_steps else d

        # add epi ret
        epi_ret += r

        # add epi cost  
        epi_cost += info["cost"]

        # memorize
        if c.Env.name == "SRL_Traj_Env":
            agent.memorize(s, a, r, s2, d)
        else:
            agent.memorize(s, a, r, s2, d)

        # LSTM: update history
        if agent.needs_history:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift=-1, axis=0)
                s_hist[agent.history_length - 1, :] = s

                a_hist = np.roll(a_hist, shift=-1, axis=0)
                a_hist[agent.history_length - 1, :] = a
            else:
                s_hist[hist_len] = s
                a_hist[hist_len] = a
                hist_len += 1

        # train
        if (total_steps >= c.upd_start_step) and (total_steps % c.upd_every == 0):
            agent.train()

        # s becomes s2
        s = s2


        # end of episode handling
        if d or (epi_steps == c.Env.max_episode_steps):

            # reset noise after episode
            if hasattr(agent, "noise"):
                agent.noise.reset()

            # LSTM: reset history
            if agent.needs_history:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, agent.num_actions))
                hist_len = 0

            # reset to initial state
            if c.Env.name == "SRL_Traj_Env":
                s, info = env.reset()
            else:
                s = env.reset()

            # log episode return
            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.store(**{f"Epi_Ret_{i}" : epi_ret[i].item()})
            else:
                agent.logger.store(Epi_Ret=epi_ret)
                agent.logger.store(Epi_Cost=epi_cost)

            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
            epi_cost = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

        # end of epoch handling
        if (total_steps + 1) % c.epoch_length == 0 and (total_steps + 1) > c.upd_start_step:

            epoch = (total_steps + 1) // c.epoch_length

            # evaluate agent with deterministic policy
            eval_ret, eval_cost, eval_cost_per_step = evaluate_policy(test_env=test_env, agent=agent, c=c)

            if agent.is_multi:
                for ret_list in eval_ret:
                    for i in range(agent.N_agents):
                        agent.logger.store(**{f"Eval_ret_{i}" : ret_list[i].item()})
            else:
                for ret in eval_ret:
                    agent.logger.store(Eval_ret=ret)
                    agent.logger.store(Eval_cost=eval_cost)
                    agent.logger.store(Eval_cost_per_step=eval_cost_per_step)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)

            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.log_tabular(f"Epi_Ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Eval_ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Q_val_{i}", average_only=True)
                    agent.logger.log_tabular(f"Critic_loss_{i}", average_only=True)
                    agent.logger.log_tabular(f"Actor_loss_{i}", average_only=True)
            else:
                agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
                agent.logger.log_tabular("Epi_Cost", with_min_and_max=True)
                agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
                agent.logger.log_tabular("Eval_cost", with_min_and_max=True)
                agent.logger.log_tabular("Q_val", with_min_and_max=True)
                agent.logger.log_tabular("Critic_loss", average_only=True)
                agent.logger.log_tabular("Actor_loss", average_only=True)

            if agent.needs_history:
                agent.logger.log_tabular("Actor_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Actor_ExtMemory", with_min_and_max=False)
                agent.logger.log_tabular("Critic_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Critic_ExtMemory", with_min_and_max=False)

            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir     = agent.logger.output_dir,
                            alg     = agent.name,
                            env_str = c.Env.name,
                            info    = c.Env.info)
            # save weights
            save_weights(agent, eval_ret)

def save_weights(agent: _Agent, eval_ret) -> None:

    # check whether this was the best evaluation epoch so far
    with open(f"{agent.logger.output_dir}/progress.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.astype(float)

    # no best-weight-saving for multi-agent problems since the definition of best weights is not straightforward anymore
    if agent.is_multi:
        best_weights = False
    elif len(df["Avg_Eval_ret"]) == 1:
        best_weights = True
    else:
        if np.mean(eval_ret) > max(df["Avg_Eval_ret"][:-1]):
            best_weights = True
        else:
            best_weights = False

    # usual save
    torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
    torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")

    # best save
    if best_weights:
        torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_best_weights.pth")
        torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_best_weights.pth")


def test(config_path, env, agent_name: str):
    c = ConfigFile(config_path)
    seed = 5600
    date = "2025-06-12"

    # get state shape
    c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "test"
    c.num_actions = 3

    # seeding
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # agent prep
    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"
    
    c.actor_weights  = f"experiments/{agent_name}_{c.Env.name}__{date}_{seed}/{agent_name}_actor_best_weights.pth"
    c.critic_weights = f"experiments/{agent_name}_{c.Env.name}__{date}_{seed}/{agent_name}_critic_best_weights.pth"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent


    # visualization
    visualize_policy(env=env, agent=agent, c=c)

def main(algo, train_mode, cost_fn_mode):

    env = make_real_no_cost_env(cost_fn_mode)
    


    if train_mode:
        
        train(
            config_path="Metadrive.yaml",
            env=env,
            agent_name=algo,
        )
    else:
        test(
            config_path="Metadrive.yaml",
            env=env,
            agent_name=algo,
        )

from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="Which algorithm use to train", default='SAC')
    parser.add_argument("-l", "--learning_rate", help="Learning rate", default='fixed')
    parser.add_argument("-m", "--mode", help="If train or test", default='train')
    parser.add_argument("-c", "--cost_fn_mode", help="Cost function mode", default='standard')
    args = parser.parse_args()

    algo = args.algorithm
    learning_rate = args.learning_rate
    train_mode = True if args.mode == "train" else False
    cost_fn_mode = args.cost_fn_mode
 
    main(algo, train_mode, cost_fn_mode)

    