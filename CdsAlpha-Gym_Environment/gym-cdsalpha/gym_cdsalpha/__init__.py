from gym.envs.registration import register

register(
    id='cdsalpha-v1',
    entry_point='gym_cdsalpha.envs:CdsalphaEnv',
)