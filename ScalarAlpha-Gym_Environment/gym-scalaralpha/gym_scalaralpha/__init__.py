from gym.envs.registration import register

register(
    id='scalaralpha-v0',
    entry_point='gym_cdsalpha.envs:CdsalphaEnv',
)