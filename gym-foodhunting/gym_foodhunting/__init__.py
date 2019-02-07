from gym.envs.registration import register
from gym_foodhunting.foodhunting.gym_foodhunting import R2D2Simple, R2D2Discrete, HSR, HSRSimple, HSRDiscrete

# FoodHunting R2D2
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': R2D2Simple, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': R2D2Simple, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': R2D2Discrete, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': R2D2Discrete, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

# FoodHunting HSR
register(
    id='FoodHuntingHSR-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': HSRSimple, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': HSRSimple, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSR-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=10.0,
    kwargs={'render': False, 'robot_model': HSRSimple, 'max_steps': 500, 'num_foods': 10, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRGUI-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=10.0,
    kwargs={'render': True, 'robot_model': HSRSimple, 'max_steps': 500, 'num_foods': 10, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': False, 'robot_model': HSRDiscrete, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=3.0,
    kwargs={'render': True, 'robot_model': HSRDiscrete, 'max_steps': 500, 'num_foods': 3, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRTestGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=500,
    reward_threshold=1.0,
    kwargs={'render': True, 'robot_model': HSR, 'max_steps': 500, 'num_foods': 1, 'food_size': 0.5, 'food_radius_scale': 1.0, 'food_angle_scale': 1.0}
)
