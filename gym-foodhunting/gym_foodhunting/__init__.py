from gym.envs.registration import register
from gym_foodhunting.foodhunting.gym_foodhunting import R2D2, R2D2Discrete, HSR, HSRDiscrete

# FoodHunting R2D2
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robotModel': R2D2, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robotModel': R2D2, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robotModel': R2D2Discrete, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robotModel': R2D2Discrete, 'max_steps': 100, 'num_foods': 3}
)

# FoodHunting HSR
register(
    id='FoodHuntingHSR-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robotModel': HSR, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingHSRGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robotModel': HSR, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingHSRDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robotModel': HSRDiscrete, 'max_steps': 100, 'num_foods': 3}
)

register(
    id='FoodHuntingHSRDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robotModel': HSRDiscrete, 'max_steps': 100, 'num_foods': 3}
)
