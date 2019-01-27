from gym.envs.registration import register

# FoodHunting
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    kwargs={'render': False, 'discrete': True}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    kwargs={'render': True, 'discrete': True}
)

register(
    id='FoodHuntingContinuous-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    kwargs={'render': False, 'discrete': False}
)

register(
    id='FoodHuntingContinuousGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    kwargs={'render': True, 'discrete': False}
)
