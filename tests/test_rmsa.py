import logging
import os
import pickle
import json
import gym
import matplotlib.pyplot as plt
import numpy as np

from optical_rl_gym.envs.rmsa_env import (
    SimpleMatrixObservation,
    least_loaded_path_first_fit,
    shortest_available_path_first_fit,
    shortest_path_first_fit,
    least_congested_path_first_fit,
    least_congested_path_KSP_first_fit,
)
from optical_rl_gym.utils import evaluate_heuristic, random_policy

output = {}
with open("rmsa_results_EU_5ksp_least_congested.json", "w") as file:
    # Load the existing JSON data
    json.dump(output, file)

for load in range(700, 801, 50):
#load = 300

    logging.getLogger("rmsaenv").setLevel(logging.INFO)

    seed = 50
    episodes = 1
    episode_length = 1000

    monitor_files = []
    policies = []

    # topology_name = 'gbn'
    # topology_name = 'nobel-us'
    # topology_name = 'germany50'
    # with open(
    #     os.path.join( "examples", "topologies", "nsfnet_chen_5-paths_6-modulations.h5"), "rb"
    # ) as f:
    #     topology = pickle.load(f)
    output.update({load:{}})
    for seeds_cnt in range(seed):
        with open(
            os.path.join( "examples", "topologies", "europe_network_5-paths_1-modulations.h5"), "rb"
        ) as f:
            topology = pickle.load(f)

        env_args = dict(
            topology=topology,
            seed=seeds_cnt,
            allow_rejection=True,
            load=load,
            mean_service_holding_time=3600,
            episode_length=episode_length,
            num_spectrum_resources=256,
            bit_rate_selection="discrete",
        )

        print("STR".ljust(5), "REW".rjust(7), "STD".rjust(7))
        seeds = [seeds_cnt]#list(range(0,episodes,1))
        init_env = gym.make("RMSA-v0", **env_args)
        env_rnd = SimpleMatrixObservation(init_env)
        mean_reward_rnd, std_reward_rnd, blocking, BW_blocking, info = evaluate_heuristic(
            env_rnd, least_congested_path_KSP_first_fit, n_eval_episodes=episodes, seeds = seeds
        )
        output[load].update({seeds_cnt:{
                        'request_blocking':blocking,
                        'BW_Blocking': BW_blocking,
                        'info': info
        }})
        with open("rmsa_results_EU_5ksp_least_congested.json", "w") as file:
        # Write the updated data back to the file
            json.dump(output, file, indent=4)
    
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
print(init_env.topology.graph["throughput"])

print("STR".ljust(5), "REW".rjust(7), "STD".rjust(7))

init_env = gym.make("RMSA-v0", **env_args)
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
print(init_env.topology.graph["throughput"])
# exit(0)

env_sp = gym.make("RMSA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_sp, shortest_path_first_fit, n_eval_episodes=episodes
)
print("SP-FF:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
print(
    "\tBit rate blocking:",
    (env_sp.episode_bit_rate_requested - env_sp.episode_bit_rate_provisioned)
    / env_sp.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_sp.episode_services_processed - env_sp.episode_services_accepted)
    / env_sp.episode_services_processed,
)

env_sap = gym.make("RMSA-v0", **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(
    env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes
)
print("SAP-FF:".ljust(8), f"{mean_reward_sap:.4f}  {std_reward_sap:.4f}")
print(
    "\tBit rate blocking:",
    (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned)
    / env_sap.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_sap.episode_services_processed - env_sap.episode_services_accepted)
    / env_sap.episode_services_processed,
)

env_llp = gym.make("RMSA-v0", **env_args)
mean_reward_llp, std_reward_llp = evaluate_heuristic(
    env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes
)
print("LLP-FF:".ljust(8), f"{mean_reward_llp:.4f}  {std_reward_llp:.4f}")
print(
    "\tBit rate blocking:",
    (env_llp.episode_bit_rate_requested - env_llp.episode_bit_rate_provisioned)
    / env_llp.episode_bit_rate_requested,
)
print(
    "\tRequest blocking:",
    (env_llp.episode_services_processed - env_llp.episode_services_accepted)
    / env_llp.episode_services_processed,
)
