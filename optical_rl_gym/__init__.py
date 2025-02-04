from gym.envs.registration import register

register(
    id="RMSA-v0",
    entry_point="optical_rl_gym.envs:RMSAEnv",
)

register(
    id="DeepRMSA-v0",
    entry_point="optical_rl_gym.envs:DeepRMSAEnv",
)

register(
    id="RWA-v0",
    entry_point="optical_rl_gym.envs:RWAEnv",
)

register(
    id="QoSConstrainedRA-v0",
    entry_point="optical_rl_gym.envs:QoSConstrainedRA",
)

register(
    id="RMCSA-v0",
    entry_point="optical_rl_gym.envs:RMCSAEnv",
)

register(
    id="RMSA-QL-v0",
    entry_point="optical_rl_gym.envs:RMSAQLEnv",
)

register(
    id="RLRMCSA-v0",
    entry_point="optical_rl_gym.envs:RLRMCSAEnv",
)