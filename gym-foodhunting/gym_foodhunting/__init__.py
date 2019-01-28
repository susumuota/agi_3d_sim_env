from gym.envs.registration import register
from gym_foodhunting.foodhunting.gym_foodhunting import HSR, R2D2

# FoodHunting HSR
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'discrete': False, 'robotModel': HSR}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'discrete': False, 'robotModel': HSR}
)

register(
    id='FoodHuntingDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'discrete': True, 'robotModel': HSR}
)

register(
    id='FoodHuntingDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'discrete': True, 'robotModel': HSR}
)

# FoodHunting R2D2
register(
    id='FoodHuntingR2D2-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'discrete': False, 'robotModel': R2D2}
)

register(
    id='FoodHuntingR2D2GUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'discrete': False, 'robotModel': R2D2}
)

register(
    id='FoodHuntingR2D2Discrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'discrete': True, 'robotModel': R2D2}
)

register(
    id='FoodHuntingR2D2DiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'discrete': True, 'robotModel': R2D2}
)
