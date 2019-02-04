from gym.envs.registration import register
from gym_foodhunting.foodhunting.gym_foodhunting import R2D2, R2D2Discrete, HSR, HSRDiscrete

# FoodHunting R2D2
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': R2D2, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': R2D2, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': R2D2Discrete, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': R2D2Discrete, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

# FoodHunting HSR
register(
    id='FoodHuntingHSR-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': HSR, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingHSRGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': HSR, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingHSR-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=200,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': HSR, 'max_steps': 200, 'num_foods': 3, 'food_size': 0.5, 'bullet_steps': 100}
)

register(
    id='FoodHuntingHSRGUI-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=200,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': HSR, 'max_steps': 200, 'num_foods': 3, 'food_size': 0.5, 'bullet_steps': 100}
)

register(
    id='FoodHuntingHSRDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': HSRDiscrete, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

register(
    id='FoodHuntingHSRDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=100,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': HSRDiscrete, 'max_steps': 100, 'num_foods': 3, 'food_size': 1.0, 'bullet_steps': 200}
)

