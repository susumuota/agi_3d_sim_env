# gym-foodhunting

Gym environments and agents for food hunting in the 3D World.


# Install

I've tested on Mac OS X 10.13.6 (Python 3.6.5) and Ubuntu 16.04.

See also these pages for more details of installation.
https://github.com/openai/gym#installation
https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
https://github.com/hill-a/stable-baselines#installation

```
git clone git@github.com:susumuota/agi_3d_sim_env.git
cd agi_3d_sim_env

python3 -m venv venv
source venv/bin/activate

pip install numpy
pip install gym
pip install pybullet
pip install stable-baselines

git clone git@github.com:ToyotaResearchInstitute/hsr_description.git
git clone git@github.com:ToyotaResearchInstitute/hsr_meshes.git

cd gym-foodhunting
pip install -e .
cd ..
```


# Uninstall

```
pip uninstall gym-foodhunting

pip uninstall stable-baselines
pip uninstall pybullet
pip uninstall gym
pip uninstall numpy

# or just remove venv directory.
```

# Example

## simplest example

```python
import gym
import gym.spaces
import gym_foodhunting

import pybullet as p

def getAction():
    keys = p.getKeyboardEvents()
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        return 0
    elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        return 1
    elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        return 2
    else:
        return 0

def main():
    env = gym.make('FoodHuntingGUI-v0')
    # env = gym.make('FoodHunting-v0')
    print(env.observation_space, env.action_space)
    obs = env.reset()
    while True:
        action = getAction()
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(action, obs, reward, done, info)
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    main()
```


# Available Environments

```
FoodHunting-v0
FoodHuntingGUI-v0
FoodHuntingContinuous-v0
FoodHuntingContinuousGUI-v0
```

# Author

Susumu OTA
