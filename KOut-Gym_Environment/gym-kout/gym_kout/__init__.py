from gym.envs.registration import register

register(
    id='kout-v0',
    entry_point='gym_kout.envs:KoutEnv',
)