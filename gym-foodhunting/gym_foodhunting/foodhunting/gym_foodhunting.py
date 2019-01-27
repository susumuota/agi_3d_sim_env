# -*- coding: utf-8 -*-

# Copyright 2019 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import gym.spaces
import gym.utils
import pybullet as p
import pybullet_data


class Robot:
    JOINT_TYPE_NAMES = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED']

    # projectionMatrix settings
    CAMERA_PIXEL_WIDTH = 64 # 64 is minimum for stable-baselines
    CAMERA_PIXEL_HEIGHT = 64 # 64 is minimum for stable-baselines
    CAMERA_FOV = 90.0
    CAMERA_NEAR_PLANE = 0.01
    CAMERA_FAR_PLANE = 100.0

    # viewMatrix settings
    # default values are for R2D2 model
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.2
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    MAX_VELOCITY = 5.0

    def __init__(self, urdfPath, position=[0.0, 0.0, 1.0], orientation=[0.0, 0.0, 0.0, 1.0]):
        self.urdfPath = urdfPath
        self.robotId = p.loadURDF(urdfPath, basePosition=position, baseOrientation=orientation)
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.CAMERA_FOV, float(self.CAMERA_PIXEL_WIDTH)/float(self.CAMERA_PIXEL_HEIGHT), self.CAMERA_NEAR_PLANE, self.CAMERA_FAR_PLANE);

    def setWheels(self, left, right):
        self.setWheelsVelocity(self.MAX_VELOCITY * left, self.MAX_VELOCITY * right)
        # self.setWheelsPosition(left, right)
        # self.setWheelsTorque(left, right)

    def setWheelsVelocity(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def setWheelsPosition(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def setWheelsTorque(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def stop(self):
        self.setWheels(0.0, 0.0)

    def forward(self, intensity=1.0):
        self.setWheels(intensity, intensity)

    def backward(self, intensity=1.0):
        self.setWheels(-intensity, -intensity)

    def turnLeft(self, intensity=1.0):
        self.setWheels(-intensity, intensity)

    def turnRight(self, intensity=1.0):
        self.setWheels(intensity, -intensity)

    def getPositionAndOrientation(self):
        return p.getBasePositionAndOrientation(self.robotId)

    def isContact(self, bodyId):
        cps = p.getContactPoints(self.robotId, bodyId)
        return len(cps) > 0

    def getCameraImage(self):
        # compute eye and target position for viewMatrix
        pos, orn, _, _, _, _ = p.getLinkState(self.robotId, self.CAMERA_JOINT_INDEX)
        cameraPos = np.array(pos)
        cameraMat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3).T
        eyePos = cameraPos + self.CAMERA_EYE_SCALE * cameraMat[self.CAMERA_EYE_INDEX]
        targetPos = cameraPos + self.CAMERA_TARGET_SCALE * cameraMat[self.CAMERA_EYE_INDEX]
        up = self.CAMERA_UP_SCALE * cameraMat[self.CAMERA_UP_INDEX]
        p.addUserDebugLine(eyePos, targetPos, lineColorRGB=[1, 0, 0], lifeTime=0.1) # red line for camera vector
        p.addUserDebugLine(eyePos, eyePos + up * 0.5, lineColorRGB=[0, 0, 1], lifeTime=0.1) # blue line for up vector
        viewMatrix = p.computeViewMatrix(eyePos, targetPos, up)
        image = p.getCameraImage(self.CAMERA_PIXEL_WIDTH, self.CAMERA_PIXEL_HEIGHT, viewMatrix, self.projectionMatrix, shadow=1, lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return image

    def getCameraObservation(self):
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 4), dtype=np.float32)
        width, height, rgbPixels, depthPixels, segmentationMaskBuffer = self.getCameraImage()
        rgba = np.array(rgbPixels, dtype=np.float32).reshape((height, width, 4))
        depth = np.array(depthPixels, dtype=np.float32).reshape((height, width, 1))
        rgb = np.delete(rgba, [3], axis=2) # delete alpha channel
        rgb01 = np.clip(rgb * 0.00392156862, 0.0, 1.0) # rgb / 255.0
        obs = np.insert(rgb01, [3], np.clip(depth, 0.0, 1.0), axis=2)
        # self.observation_space = gym.spaces.Box(low=-1.0, high=255.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 6), dtype=np.float32)
        # width, height, rgbPixels, depthPixels, segmentationMaskBuffer = self.getCameraImage()
        # rgba = np.array(rgbPixels, dtype=np.float32).reshape((height, width, 4))
        # depth = np.array(depthPixels, dtype=np.float32).reshape((height, width, 1))
        # seg = np.array(segmentationMaskBuffer, dtype=np.float32).reshape((height, width, 1))
        # obs = np.insert(rgba, [4], depth, axis=2)
        # obs = np.insert(obs, [5], seg, axis=2)
        return obs

    def printJointInfo(self, index):
        jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex = p.getJointInfo(self.robotId, index)
        line = [ jointName.decode('ascii'), '\n\tjointIndex\t', jointIndex, '\n\tjointName\t', jointName, '\n\tjointType\t', jointType, '\t', self.JOINT_TYPE_NAMES[jointType], '\n\tqIndex\t', qIndex, '\n\tuIndex\t', uIndex, '\n\tflags\t', flags, '\n\tjointDamping\t', jointDamping, '\n\tjointFriction\t', jointFriction, '\n\tjointLowerLimit\t', jointLowerLimit, '\n\tjointUpperLimit\t', jointUpperLimit, '\n\tjointMaxForce\t', jointMaxForce, '\n\tjointMaxVelocity\t', jointMaxVelocity, '\n\tlinkName\t', linkName, '\n\tjointAxis\t', jointAxis, '\n\tparentFramePos\t', parentFramePos, '\n\tparentFrameOrn\t', parentFrameOrn, '\n\tparentIndex\t', parentIndex, '\n' ]
        print(''.join([ str(item) for item in line ]))
        #line = [ jointIndex, jointName.decode('ascii'), self.JOINT_TYPE_NAMES[jointType] ]
        #print('\t'.join([ str(item) for item in line ]))

    def printJointInfoArray(self, indexArray):
        for index in indexArray:
            self.printJointInfo(index)

    def printAllJointInfo(self):
        self.printJointInfoArray(range(p.getNumJoints(self.robotId)))


class HSR(Robot):
    HSR_URDF_PATH = '/Users/ota/Documents/python/bullet/hsr_description/robots/hsrb4s.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 19
    CAMERA_EYE_INDEX = 2
    CAMERA_UP_INDEX = 1
    CAMERA_EYE_SCALE = 0.15
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = -1.0

    def __init__(self, urdfPath=HSR_URDF_PATH, position=[0.0, 0.0, 0.05], orientation=[0.0, 0.0, 0.0, 1.0]):
        super(HSR, self).__init__(urdfPath, position, orientation)

    # override methods
    def setWheelsVelocity(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[right, left])

    def setWheelsPosition(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[right, left])

    def setWheelsTorque(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[right, left])

    # HSR specific methods
    def setArmPosition(self, lift, flex, roll):
        # lift:  0.00 to 0.69
        # flex: -2.62 to 0.00
        # roll: -2.09 to 3.84
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[23, 24, 25],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[lift, flex, roll])

    def setWristPosition(self, flex, roll):
        # flex: -1.92 to 1.22
        # roll: -1.92 to 3.67
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[26, 27],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[flex, roll])

class R2D2(Robot):
    R2D2_URDF_PATH = 'r2d2.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.2
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    def __init__(self, urdfPath=R2D2_URDF_PATH, position=[0.0, 0.0, 0.5], orientation=[0.0, 0.0, 0.0, 1.0]):
        super(R2D2, self).__init__(urdfPath, position, orientation)

    # override methods
    def setWheelsVelocity(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[-right, -right, -left, -left])

    def setWheelsPosition(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-right, -right, -left, -left])
    def setWheelsTorque(self, left, right):
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[-right, -right, -left, -left])

class FoodHuntingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    GRAVITY = -10.0
    BULLET_STEPS_PER_GYM_STEP = 200
    MAX_STEPS = 100
    NUM_FOODS = 3

    def __init__(self, render=False, discrete=False):
        ### gym variables
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 4), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=-1.0, high=255.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 6), dtype=np.float32)
        self.isDiscrete = discrete
        if self.isDiscrete:
            self.action_space = gym.spaces.Discrete(3) # 0, 1, 2
        else:
            self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.reward_range = (0.0, 1.0)
        # self.seed()
        ### pybullet settings
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = None
        self.robot = None
        self.foodIds = []
        ### episode variables
        self.steps = 0
        self.episode_rewards = 0.0

    def close(self):
        p.disconnect(self.physicsClient)

    def reset(self):
        self.steps = 0
        self.episode_rewards = 0.0
        p.resetSimulation()
        # p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, self.GRAVITY)
        self.planeId = p.loadURDF('plane.urdf')
        #self.robot = R2D2()
        self.robot = HSR()
        self.foodIds = []
        for foodPos in self._generateFoodPositions(self.NUM_FOODS):
            foodId = p.loadURDF('sphere2red.urdf', foodPos, globalScaling=1.0)
            self.foodIds.append(foodId)
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
        obs = self.robot.getCameraObservation()
        return obs

    def step(self, action):
        self._applyAction(action)
        self.steps += 1
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
        contactedFoodIds = [ foodId for foodId in self.foodIds if self.robot.isContact(foodId) ]
        for foodId in contactedFoodIds:
            p.removeBody(foodId)
            self.foodIds.remove(foodId)
        obs = self.robot.getCameraObservation()
        reward = len(contactedFoodIds) * 1.0
        self.episode_rewards += reward
        done = self.steps >= self.MAX_STEPS or len(self.foodIds) <= 0
        robotPos, robotOrn = self.robot.getPositionAndOrientation()
        info = { 'steps': self.steps, 'pos': robotPos, 'orn': robotOrn }
        if done:
            info['episode'] = { 'r': self.episode_rewards, 'l': self.steps }
            print(self.episode_rewards, self.steps)
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        _, seed = np.utils.seeding.np_random(seed)
        return [seed]

    def _applyAction(self, action):
        if self.isDiscrete:
            self._applyActionDiscrete(action)
        else:
            self._applyActionContinuous(action)

    def _applyActionDiscrete(self, action):
        if action == 0:
            self.robot.forward()
        elif action == 1:
            self.robot.turnLeft()
        elif action == 2:
            self.robot.turnRight()
        else:
            raise ValueError

    def _applyActionContinuous(self, action):
        self.robot.setWheels(action[0], action[1])

    def _generateFoodPositions(self, n):
        # TODO: parameterize
        def genPos():
            r = 1.0 * np.random.rand() + 1.0
            ang = 2.0 * np.pi * np.random.rand()
            return np.array([r * np.sin(ang), r * np.cos(ang), 1.5])
        def isNear(pos, poss):
            for p in poss:
                if np.linalg.norm(p - pos) < 1.0: # too close
                    return True
            return False
        def genPosRetry(poss):
            for i in range(10):
                pos = genPos()
                if not isNear(pos, poss):
                    return pos
            return genPos()
        poss = []
        for i in range(n):
            pos = genPosRetry(poss)
            poss.append(pos)
        return poss
