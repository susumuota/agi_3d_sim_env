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
from gym.utils import seeding
import pybullet as p
import pybullet_data


class Robot:
    # default values are for R2D2 model
    URDF_PATH = 'r2d2.urdf'

    # projectionMatrix settings
    CAMERA_PIXEL_WIDTH = 64 # 64 is minimum for stable-baselines
    CAMERA_PIXEL_HEIGHT = 64 # 64 is minimum for stable-baselines
    CAMERA_FOV = 90.0
    CAMERA_NEAR_PLANE = 0.01
    CAMERA_FAR_PLANE = 100.0

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.05
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    # for debug
    JOINT_TYPE_NAMES = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED']

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 1.0], orientation=[0.0, 0.0, 0.0, 1.0]):
        self.urdfPath = urdfPath
        self.robotId = p.loadURDF(urdfPath, basePosition=position, baseOrientation=orientation)
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.CAMERA_FOV, float(self.CAMERA_PIXEL_WIDTH)/float(self.CAMERA_PIXEL_HEIGHT), self.CAMERA_NEAR_PLANE, self.CAMERA_FAR_PLANE);

    @classmethod
    def getObservationSpace(cls):
        return gym.spaces.Box(low=0.0, high=1.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 4), dtype=np.float32)

    @classmethod
    def getActionSpace(cls):
        # raise NotImplementedError
        n = 2
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        raise NotImplementedError

    def scaleJointVelocity(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxVelocity = abs(info[11])
        value *= maxVelocity
        value = -maxVelocity if value < -maxVelocity else value
        value = maxVelocity if value > maxVelocity else value
        return value

    def setJointVelocity(self, jointIndex, value, scale=1.0):
        # value should be from -1.0 to 1.0
        value = self.scaleJointVelocity(jointIndex, value)
        value *= scale
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=value)

    def scaleJointPosition(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        lowerLimit = info[8]
        upperLimit = info[9]
        maxVelocity = abs(info[11])
        if lowerLimit > upperLimit:
            lowerLimit, upperLimit = upperLimit, lowerLimit # swap
        # value *= max(abs(lowerLimit), abs(upperLimit)) # TODO: is it OK?
        # y - l = a (x - -1) = a (x + 1)
        # a = (u - l) / (1 - -1) = (u - l) / 2
        # y - l = (u - l) (x + 1) / 2
        # y = (u - l) (x + 1) * 0.5 + l
        value = (upperLimit - lowerLimit) * (value + 1.0) * 0.5 + lowerLimit
        value = lowerLimit if value < lowerLimit else value
        value = upperLimit if value > upperLimit else value
        return value, maxVelocity

    def setJointPosition(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        value, maxVelocity = self.scaleJointPosition(jointIndex, value)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=maxVelocity,
                                targetPosition=value)

    def invScaleJointPosition(self, jointIndex, value):
        info = p.getJointInfo(self.robotId, jointIndex)
        lowerLimit = info[8]
        upperLimit = info[9]
        # y - -1 = a (x - l)
        # a = (1 - -1) / (u - l) = 2 / (u - l)
        # y - -1 = 2 (x - l) / (u - l)
        # y = 2 (x - l) / (u - l) - 1
        value = 2.0 * (value - lowerLimit) / (upperLimit - lowerLimit) - 1.0
        # value should be from -1.0 to 1.0
        value = -1.0 if value < -1.0 else value
        value =  1.0 if value >  1.0 else value
        return value

    def getJointPosition(self, jointIndex):
        jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque = p.getJointState(self.robotId, jointIndex)
        return self.invScaleJointPosition(jointIndex, jointPosition)

    def scaleJointForce(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxForce = abs(info[10])
        value *= maxForce
        value = -maxForce if value < -maxForce else value
        value = maxForce if value > maxForce else value
        return value

    def setJointForce(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        value = self.scaleJointForce(jointIndex, value)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=jointIndex,
                                controlMode=p.TORQUE_CONTROL,
                                force=value)

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

    def getObservation(self):
        width, height, rgbPixels, depthPixels, segmentationMaskBuffer = self.getCameraImage()
        rgba = np.array(rgbPixels, dtype=np.float32).reshape((height, width, 4))
        depth = np.array(depthPixels, dtype=np.float32).reshape((height, width, 1))
        # seg = np.array(segmentationMaskBuffer, dtype=np.float32).reshape((height, width, 1))
        rgb = np.delete(rgba, [3], axis=2) # delete alpha channel
        rgb01 = np.clip(rgb * 0.00392156862, 0.0, 1.0) # rgb / 255.0, normalize
        obs = np.insert(rgb01, [3], np.clip(depth, 0.0, 1.0), axis=2)
        # obs = np.insert(obs, [4], seg, axis=2) # TODO: normalize
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
    URDF_PATH = 'hsrb4s.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 19
    CAMERA_EYE_INDEX = 2
    CAMERA_UP_INDEX = 1
    CAMERA_EYE_SCALE = 0.01
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = -1.0

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.05], orientation=[0.0, 0.0, 0.0, 1.0]):
        super(HSR, self).__init__(urdfPath, position, orientation)

    # override methods
    @classmethod
    def getActionSpace(cls):
        n = 4
        # n = 20
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # def setAction(self, action):
    #     self.setWheelVelocity(action[0], action[1])
    #     self.setBaseRollPosition(action[2])
    #     self.setTorsoLiftPosition(action[3])
    #     self.setHeadPosition(action[4], action[5])
    #     self.setArmPosition(action[6], action[7], action[8])
    #     self.setWristPosition(action[9], action[10])
    #     self.setHandPosition(action[11], action[12], action[13], action[14], action[15], action[16], action[17], action[18], action[19])

    def setAction(self, action):
        self.setWheelVelocity(action[0], action[1])
        self.setArmPosition(action[2], action[3], 0.0)

    # HSR specific methods
    def setWheelVelocity(self, left, right):
        self.setJointVelocity(2, right, 0.25)
        self.setJointVelocity(3, left, 0.25)

    def setBaseRollPosition(self, roll):
        self.setJointPosition(1, roll)

    def setTorsoLiftPosition(self, lift):
        self.setJointPosition(12, lift)

    def setHeadPosition(self, pan, tilt):
        self.setJointPosition(13, pan)
        self.setJointPosition(14, tilt)

    def setArmPosition(self, lift, flex, roll):
        self.setJointPosition(23, lift)
        self.setJointPosition(24, flex)
        # self.setJointPosition(25, roll)

    def setWristPosition(self, flex, roll):
        self.setJointPosition(26, flex)
        self.setJointPosition(27, roll)

    def setHandPosition(self, motor, leftProximal, leftSpringProximal, leftMimicDistal, leftDistal, rightProximal, rightSpringProximal, rightMimicDistal, rightDistal):
        self.setJointPosition(30, motor)
        self.setJointPosition(31, leftProximal)
        self.setJointPosition(32, leftSpringProximal)
        self.setJointPosition(33, leftMimicDistal)
        self.setJointPosition(34, leftDistal)
        self.setJointPosition(37, rightProximal)
        self.setJointPosition(38, rightSpringProximal)
        self.setJointPosition(39, rightMimicDistal)
        self.setJointPosition(40, rightDistal)

class HSRDiscrete(HSR):
    ACTIONS = [ [ 1.0, 1.0], [-1.0, 1.0], [1.0, -1.0] ]

    @classmethod
    def getActionSpace(cls):
        return gym.spaces.Discrete(3) # 0, 1, 2

    def setAction(self, action):
        self.setWheelVelocity(*self.ACTIONS[action])

class R2D2(Robot):
    URDF_PATH = 'r2d2.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.05
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.5], orientation=[0.0, 0.0, 0.0, 1.0]):
        super(R2D2, self).__init__(urdfPath, position, orientation)

    # override methods
    @classmethod
    def getActionSpace(cls):
        n = 6
        # n = 3
        low = -1.0 * np.ones(n)
        high = 1.0 * np.ones(n)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        self.setWheelVelocity(action[0], action[1])
        self.setGripperPosition(action[2], action[3], action[4])
        self.setHeadPosition(action[5])
        # self.setGripperPosition(0.0, action[2], action[2])

    # R2D2 specific methods
    def setWheelVelocity(self, left, right):
        self.setJointVelocity(2, right, -0.1)
        self.setJointVelocity(3, right, -0.1)
        self.setJointVelocity(6, left, -0.1)
        self.setJointVelocity(7, left, -0.1)

    def setGripperPosition(self, extension, left, right):
        self.setJointPosition(8, extension)
        self.setJointPosition(9, left)
        self.setJointPosition(11, right)

    def setHeadPosition(self, pan):
        self.setJointPosition(13, pan)

class R2D2Discrete(R2D2):
    ACTIONS = [ [ 1.0, 1.0], [-1.0, 1.0], [1.0, -1.0] ]

    @classmethod
    def getActionSpace(cls):
        return gym.spaces.Discrete(3) # 0, 1, 2

    def setAction(self, action):
        self.setWheelVelocity(*self.ACTIONS[action])

class FoodHuntingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    GRAVITY = -10.0
    BULLET_STEPS_PER_GYM_STEP = 200

    def __init__(self, render=False, robotModel=R2D2, max_steps=100, num_foods=3):
        ### gym variables
        self.observation_space = robotModel.getObservationSpace() # classmethod
        self.action_space = robotModel.getActionSpace() # classmethod
        self.reward_range = (-1.0, 1.0)
        self.seed()
        ### pybullet settings
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotModel = robotModel
        self.num_foods = num_foods
        self.max_steps = max_steps
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
        self.episode_rewards = 0
        p.resetSimulation()
        # p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, self.GRAVITY)
        self.planeId = p.loadURDF('plane.urdf')
        self.robot = self.robotModel()
        self.foodIds = []
        for foodPos in self._generateFoodPositions(self.num_foods):
            foodId = p.loadURDF('sphere2red.urdf', foodPos, globalScaling=1.0)
            self.foodIds.append(foodId)
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
        obs = self.robot.getObservation()
        return obs

    def step(self, action):
        self.steps += 1
        self.robot.setAction(action)
        reward = -0.01 # so agent needs to eat foods quickly
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
            reward += self._getReward()
        self.episode_rewards += reward
        obs = self.robot.getObservation()
        done = self.steps >= self.max_steps or len(self.foodIds) <= 0
        robotPos, robotOrn = self.robot.getPositionAndOrientation()
        info = { 'steps': self.steps, 'pos': robotPos, 'orn': robotOrn }
        if done:
            info['episode'] = { 'r': self.episode_rewards, 'l': self.steps }
            # print(self.episode_rewards, self.steps)
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getReward(self):
        reward = 0
        contactedFoodIds = [ foodId for foodId in self.foodIds if self.robot.isContact(foodId) ]
        for foodId in contactedFoodIds:
            p.removeBody(foodId)
            self.foodIds.remove(foodId)
            reward += 1
        return reward

    def _generateFoodPositions(self, n):
        # TODO: parameterize
        def genPos():
            r = 1.0 * self.np_random.rand() + 1.0
            ang = 2.0 * np.pi * self.np_random.rand()
            return np.array([r * np.sin(ang), r * np.cos(ang), 1.5])
        def isNear(pos, poss):
            for p in poss:
                if np.linalg.norm(p - pos) < 1.0:
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
