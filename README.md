# AGI 3D Simulation Environment

A 3D simulation environment for use in human level intelligence (HLAI) and Artificial General Intelligence (AGI) research.


# Install

I've tested on Mac OS X 10.13.6 and Ubuntu 16.04.

```
mkdir myproject
cd myproject

python3 -m venv venv
source venv/bin/activate

pip install numpy
pip install gym
pip install pybullet
pip install stable-baselines

git clone git@github.com:ToyotaResearchInstitute/hsr_description.git
git clone git@github.com:ToyotaResearchInstitute/hsr_meshes.git

git clone git@github.com:susumuota/agi_3d_sim_env.git
cd agi_3d_sim_env/gym-foodhunting
pip install -e .
cd ../..
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

# Links

- Request for Reseach

https://wba-initiative.org/en/research/rfr/3d-agent-test-suites/


# Robot models

This repository can uses these robot models.

## Toyota HSR by TRI

- URDF robot models for Toyota HSR by TRI.

https://github.com/ToyotaResearchInstitute/hsr_description

- 3D Mesh files for Toyota HSR by TRI.

https://github.com/ToyotaResearchInstitute/hsr_meshes

## R2D2 by Bullet Physics

- Bullet Physics SDK

https://github.com/bulletphysics/bullet3


# Author

Susumu OTA
