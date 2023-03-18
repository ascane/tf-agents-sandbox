from gym.envs.registration import register

# https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
# https://github.com/openai/gym/blob/master/gym/envs/registration.py#L434

register(
    id='simple-sde-v0',
    entry_point='src.envs:SimpleSdeEnv'
    # tags={'wrapper_config.TimeLimit.max_episode_steps': 6060},
    # timestep_limit=6060,
    # reward_threshold=1000
)