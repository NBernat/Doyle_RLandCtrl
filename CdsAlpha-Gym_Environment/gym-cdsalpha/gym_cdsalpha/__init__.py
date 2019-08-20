from gym.envs.registration import register

register(
    id='cdsalpha-v0',
    entry_point='gym_cdsalpha.envs:CdsalphaEnv',
)