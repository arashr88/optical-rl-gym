import logging
import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from optical_rl_gym.envs.rlrmcsa_env import (
    SimpleMatrixObservation,
    shortest_available_path_best_modulation_first_core_first_fit,
    shortest_path_best_modulation_first_core_first_fit,
)
from optical_rl_gym.utils import evaluate_heuristic, random_policy
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                return obj.tolist()  # Convert float32 elements to regular float
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)  # Convert standalone float32 elements to regular float
        return json.JSONEncoder.default(self, obj)
    
    
with open(
    os.path.join(
        "examples", "topologies", "nsfnet_chen_5-paths_6-modulations.h5"
    ),
    "rb",
) as f:
    topology = pickle.load(f)
    
    
logging.getLogger("rlrmsaenv").setLevel(logging.INFO)
load = 1000
seed = 3000
episodes = 10
episode_length = 100000
num_spatial_resources = 7
worst_xt = -84.7

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'

output = {}
with open("data_XT_DQN_newObs.json", "w") as file:
    # Load the existing JSON data
    json.dump(output, file)

flag = False
output.update({load:{}})
for seeds_cnt in range(episodes):
    env_args = dict(
        topology=topology,
        seed=seeds_cnt,
        allow_rejection=False,
        load=load,
        mean_service_holding_time=3600,
        episode_length=episode_length,
        num_spectrum_resources=128,
        num_spatial_resources=num_spatial_resources,
        worst_xt=worst_xt,
    )
    seeds = [seeds_cnt]
    print("STR".ljust(5), "REW".rjust(7), "STD".rjust(7))
    print("Erlang:", load, "    seed:  ", seeds_cnt)
    env_sap = gym.make("RLRMCSA-v0", **env_args)
    env_sap = DummyVecEnv([lambda: env_sap])
    if flag == False:
        model = DQN("MlpPolicy", env_sap, verbose=1)
        model.save("deepq_XT_DQN-newObs")
        flag = True
    else:
        model = DQN.load("deepq_XT_DQN-newObs")
        model.set_env(env_sap)
    mean_reward_sap, std_reward_sap, blocking, BW_blocking, info = evaluate_heuristic(
        env_sap,
        shortest_path_best_modulation_first_core_first_fit,
        n_eval_episodes=episodes,
        seeds = seeds,
        loaded_model=model,
    )
    output[load].update({seeds_cnt:{
                            'request_blocking': blocking,
                            'BW_Blocking': BW_blocking,
                            'mean_reward': mean_reward_sap,
                            'std_reward': std_reward_sap,
                            'info' : info,

            }})
    with open("data_XT_DQN_newObs.json", "w") as file:
    # Write the updated data back to the file
        json.dump(output, file, indent=4, cls=NumpyArrayEncoder)
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('\tBit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
print('\tRequest blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
print("Throughput:", env_sap.topology.graph["throughput"])

"""
init_env = gym.make("RLRMCSA-v0", **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(
    env_rnd, random_policy, n_eval_episodes=episodes
)
print("Rnd:".ljust(8), f"{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}")
print(
    "\tBit rate blocking:",
    (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned)
    / init_env.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (init_env.episode_services_processed - init_env.episode_services_accepted)
    / init_env.episode_services_processed,
)
print("Throughput:", init_env.topology.graph["throughput"])

env_sap = gym.make("RLRMCSA-v0", **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(
    env_sap,
    shortest_available_path_best_modulation_first_core_first_fit,
    n_eval_episodes=episodes,
)
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('\tBit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
print('\tRequest blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
print("Throughput:", env_sap.topology.graph["throughput"])
#
# # Initial Metrics for Environment
# print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
# print('\tBit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
# print('\tRequest blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
#
# # Additional Metrics For Environment
# print('\tThroughput:', env_sap.topology.graph['throughput'])
# print('\tCompactness:', env_sap.topology.graph['compactness'])
# print('\tResource Utilization:', np.mean(env_sap.utilization))
# for key, value in env_sap.core_utilization.items():
#     print('\t\tUtilization per core ({}): {}'.format(key, np.mean(env_sap.core_utilization[key])))


#Specific - modify
env_sp = gym.make('RMCSA-v0', **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
print('SP-FF:'.ljust(8), f'{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}')
print('Bit rate blocking:', (env_sp.episode_bit_rate_requested - env_sp.episode_bit_rate_provisioned) / env_sp.episode_bit_rate_requested)
print('Request blocking:', (env_sp.episode_services_processed - env_sp.episode_services_accepted) / env_sp.episode_services_processed)

env_sap = gym.make('RMCSA-v0', **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('Bit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
print('Request blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)

env_llp = gym.make('RMCSA-v0', **env_args)
mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
print('LLP-FF:'.ljust(8), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
print('Bit rate blocking:', (env_llp.episode_bit_rate_requested - env_llp.episode_bit_rate_provisioned) / env_llp.episode_bit_rate_requested)
print('Request blocking:', (env_llp.episode_services_processed - env_llp.episode_services_accepted) / env_llp.episode_services_processed)

"""
