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

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 1.0], orientation=[0.0, 0.0, 0.0, 1.0], isDiscreteAction=False):
        self.urdfPath = urdfPath
        self.robotId = p.loadURDF(urdfPath, basePosition=position, baseOrientation=orientation)
        self.isDiscreteAction = isDiscreteAction
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.CAMERA_FOV, float(self.CAMERA_PIXEL_WIDTH)/float(self.CAMERA_PIXEL_HEIGHT), self.CAMERA_NEAR_PLANE, self.CAMERA_FAR_PLANE);

    @classmethod
    def getActionSpace(cls, isDiscreteAction):
        if isDiscreteAction:
            return gym.spaces.Discrete(3) # 0, 1, 2
        else:
            n = 2
            low = -1.0 * np.ones(n)
            high = 1.0 * np.ones(n)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def getObservationSpace(cls):
        return gym.spaces.Box(low=0.0, high=1.0, shape=(Robot.CAMERA_PIXEL_HEIGHT, Robot.CAMERA_PIXEL_WIDTH, 4), dtype=np.float32)

    def setAction(self, action):
        if self.isDiscreteAction:
            if action == 0:
                self.setWheelVelocity(1.0, 1.0) # forward
            elif action == 1:
                self.setWheelVelocity(-1.0, 1.0) # turn left
            elif action == 2:
                self.setWheelVelocity(1.0, -1.0) # turn right
            else:
                raise ValueError
        else:
            self.setWheelVelocity(action[0], action[1])

    def setWheelVelocity(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def setWheelPosition(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def setWheelTorque(self, left, right):
        # implement this in subclass
        raise NotImplementedError

    def scaleJointVelocity(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxVelocity = abs(info[11])
        value *= maxVelocity
        value = -maxVelocity if value < -maxVelocity else value
        value = maxVelocity if value > maxVelocity else value
        return value

    def scaleJointPosition(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        lowerLimit = info[8]
        upperLimit = info[9]
        maxVelocity = abs(info[11])
        if lowerLimit > upperLimit:
            lowerLimit, upperLimit = upperLimit, lowerLimit # swap
        value *= max(abs(lowerLimit), abs(upperLimit)) # TODO: is it OK?
        value = lowerLimit if value < lowerLimit else value
        value = upperLimit if value > upperLimit else value
        return value, maxVelocity

    def scaleJointForce(self, jointIndex, value):
        # value should be from -1.0 to 1.0
        info = p.getJointInfo(self.robotId, jointIndex)
        maxForce = abs(info[10])
        value *= maxForce
        value = -maxForce if value < -maxForce else value
        value = maxForce if value > maxForce else value
        return value

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

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.05], orientation=[0.0, 0.0, 0.0, 1.0], isDiscreteAction=False):
        super(HSR, self).__init__(urdfPath, position, orientation, isDiscreteAction)
        self.robot.setArmPosition(1.0, -1.0, 0.0)

    # override methods
    @classmethod
    def getActionSpace(cls, isDiscreteAction):
        if isDiscreteAction:
            return gym.spaces.Discrete(3) # 0, 1, 2
        else:
            n = 19
            low = -1.0 * np.ones(n)
            high = 1.0 * np.ones(n)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        if self.isDiscreteAction:
            if action == 0:
                self.setWheelVelocity(1.0, 1.0) # forward
            elif action == 1:
                self.setWheelVelocity(-1.0, 1.0) # turn left
            elif action == 2:
                self.setWheelVelocity(1.0, -1.0) # turn right
            else:
                raise ValueError
        else:
            self.setWheelVelocity(action[0], action[1])
            self.setBaseRollPosition(action[2])
            self.setTorsoLiftPosition(action[3])
            self.setHeadPosition(action[4], action[5])
            self.setArmPosition(action[6], action[7], action[8])
            self.setWristPosition(action[9], action[10])
            self.setHandPosition(action[11], action[12], action[13], action[14], action[15], action[16], action[17], action[18], action[19])

    def setWheelVelocity(self, left, right):
        right = self.scaleJointVelocity(2, right)
        left = self.scaleJointVelocity(3, left)
        right *= 0.25 # TODO
        left *= 0.25 # TODO
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[right, left])

    def setWheelPosition(self, left, right):
        # TODO: 
        right, rightMaxVelocity = self.scaleJointPosition(2, right)
        left, leftMaxVelocity = self.scaleJointPosition(3, left)
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[right, left])

    def setWheelTorque(self, left, right):
        # TODO:
        right = self.scaleJointForce(2, right)
        left = self.scaleJointForce(3, left)
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3], # right, left
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[right, left])

    # HSR specific methods
    def setBaseRollVelocity(self, roll):
        roll = self.scaleJointVelocity(1, roll)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=1,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=roll)

    def setBaseRollPosition(self, roll):
        orgRoll = roll
        roll, rollMaxVelocity = self.scaleJointPosition(1, roll) # TODO
        roll = orgRoll * np.pi
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rollMaxVelocity,
                                targetPosition=roll)

    def setTorsoLiftPosition(self, lift):
        # 12, torso_lift_joint
        lift, liftMaxVelocity = self.scaleJointPosition(12, lift)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=12,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=liftMaxVelocity,
                                targetPosition=lift)

    def setHeadPosition(self, pan, tilt):
        # 13, head_pan_joint
        pan, panMaxVelocity = self.scaleJointPosition(13, pan)
        # 14, head_tilt_joint
        tilt, tiltMaxVelocity = self.scaleJointPosition(14, tilt)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=13,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=panMaxVelocity,
                                targetPosition=pan)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=14,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=tiltMaxVelocity,
                                targetPosition=tilt)

    def setArmPosition(self, lift, flex, roll):
        # 23, arm_lift_joint
        lift, liftMaxVelocity = self.scaleJointPosition(23, lift)
        lift *= 0.5 # it seems wrong value...
        # 24, arm_flex_joint
        flex, flexMaxVelocity = self.scaleJointPosition(24, flex)
        # 25, arm_roll_joint
        roll, rollMaxVelocity = self.scaleJointPosition(25, roll)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=23,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=liftMaxVelocity,
                                targetPosition=lift)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=24,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=flexMaxVelocity,
                                targetPosition=flex)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=25,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rollMaxVelocity,
                                targetPosition=roll)

    def setWristPosition(self, flex, roll):
        # TODO
        # 26, wrist_flex_joint
        flex, flexMaxVelocity = self.scaleJointPosition(26, flex)
        # 27, wrist_roll_joint
        roll, rollMaxVelocity = self.scaleJointPosition(27, roll)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=26,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=flexMaxVelocity,
                                targetPosition=flex)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=27,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rollMaxVelocity,
                                targetPosition=roll)

    def setHandPosition(self, motor, leftProximal, leftSpringProximal, leftMimicDistal, leftDistal, rightProximal, rightSpringProximal, rightMimicDistal, rightDistal):
        # 30, hand_motor_joint
        motor, motorMaxVelocity = self.scaleJointPosition(30, motor)
        # 31, hand_l_proximal_joint
        leftProximal, leftProximalMaxVelocity = self.scaleJointPosition(31, leftProximal)
        # 32, hand_l_spring_proximal_joint
        leftSpringProximal, leftSpringProximalMaxVelocity = self.scaleJointPosition(32, leftSpringProximal)
        # 33, hand_l_mimic_distal_joint
        leftMimicDistal, leftMimicDistalMaxVelocity = self.scaleJointPosition(33, leftMimicDistal)
        # 34, hand_l_distal_joint
        leftDistal, leftDistalMaxVelocity = self.scaleJointPosition(34, leftDistal)
        # 37, hand_r_proximal_joint
        rightProximal, rightProximalMaxVelocity = self.scaleJointPosition(37, rightProximal)
        # 38, hand_r_spring_proximal_joint
        rightSpringProximal, rightSpringProximalMaxVelocity = self.scaleJointPosition(38, rightSpringProximal)
        # 39, hand_r_mimic_distal_joint
        rightMimicDistal, rightMimicDistalMaxVelocity = self.scaleJointPosition(39, rightMimicDistal)
        # 40, hand_r_distal_joint
        rightDistal, rightDistalMaxVelocity = self.scaleJointPosition(40, rightDistal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=30,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=motorMaxVelocity,
                                targetPosition=motor)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=31,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=leftProximalMaxVelocity,
                                targetPosition=leftProximal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=32,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=leftSpringProximalMaxVelocity,
                                targetPosition=leftSpringProximal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=33,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=leftMimicDistalMaxVelocity,
                                targetPosition=leftMimicDistal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=34,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=leftDistalMaxVelocity,
                                targetPosition=leftDistal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=37,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rightProximalMaxVelocity,
                                targetPosition=rightProximal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=38,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rightSpringProximalMaxVelocity,
                                targetPosition=rightSpringProximal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=39,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rightMimicDistalMaxVelocity,
                                targetPosition=rightMimicDistal)
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=40,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=rightDistalMaxVelocity,
                                targetPosition=rightDistal)

class R2D2(Robot):
    URDF_PATH = 'r2d2.urdf'

    # viewMatrix settings
    CAMERA_JOINT_INDEX = 14
    CAMERA_EYE_INDEX = 1
    CAMERA_UP_INDEX = 2
    CAMERA_EYE_SCALE = 0.05
    CAMERA_TARGET_SCALE = 1.0
    CAMERA_UP_SCALE = 1.0

    def __init__(self, urdfPath=URDF_PATH, position=[0.0, 0.0, 0.5], orientation=[0.0, 0.0, 0.0, 1.0], isDiscreteAction=False):
        super(R2D2, self).__init__(urdfPath, position, orientation, isDiscreteAction)

    # override methods
    @classmethod
    def getActionSpace(cls, isDiscreteAction):
        if isDiscreteAction:
            return gym.spaces.Discrete(3) # 0, 1, 2
        else:
            n = 3
            low = -1.0 * np.ones(n)
            high = 1.0 * np.ones(n)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def setAction(self, action):
        if self.isDiscreteAction:
            if action == 0:
                self.setWheelVelocity(1.0, 1.0)
            elif action == 1:
                self.setWheelVelocity(-1.0, 1.0)
            elif action == 2:
                self.setWheelVelocity(1.0, -1.0)
            else:
                raise ValueError
        else:
            self.setWheelVelocity(action[0], action[1])
            self.setHeadPosition(action[2])

    def setWheelVelocity(self, left, right):
        rf = self.scaleJointVelocity(2, right)
        rb = self.scaleJointVelocity(3, right)
        lf = self.scaleJointVelocity(6, left)
        lb = self.scaleJointVelocity(7, left)
        rf *= 0.1 # TODO
        rb *= 0.1 # TODO
        lf *= 0.1 # TODO
        lb *= 0.1 # TODO
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[-rf, -rb, -lf, -lb])

    def setWheelPosition(self, left, right):
        rf, rfMaxVelocity = self.scaleJointPosition(2, right)
        rb, rbMaxVelocity = self.scaleJointPosition(3, right)
        lf, lfMaxVelocity = self.scaleJointPosition(6, left)
        lb, lbMaxVelocity = self.scaleJointPosition(7, left)
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-rf, -rb, -lf, -lb])

    def setWheelTorque(self, left, right):
        rf = self.scaleJointForce(2, right)
        rb = self.scaleJointForce(3, right)
        lf = self.scaleJointForce(6, left)
        lb = self.scaleJointForce(7, left)
        p.setJointMotorControlArray(bodyIndex=self.robotId,
                                    jointIndices=[2, 3, 6, 7], # right front, right back, left front, left back
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[-rf, -rb, -lf, -lb])

    # R2D2 specific methods
    def setHeadPosition(self, pan):
        # 13, head_swivel
        orgPan = pan
        pan, panMaxVelocity = self.scaleJointPosition(13, pan)
        pan = orgPan
        p.setJointMotorControl2(bodyIndex=self.robotId,
                                jointIndex=13,
                                controlMode=p.POSITION_CONTROL,
                                maxVelocity=panMaxVelocity,
                                targetPosition=pan)

class FoodHuntingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    GRAVITY = -10.0
    BULLET_STEPS_PER_GYM_STEP = 200
    MAX_STEPS = 100
    NUM_FOODS = 3

    def __init__(self, render=False, discrete=False, robotModel=R2D2):
        ### gym variables
        self.isDiscreteAction = discrete
        self.observation_space = robotModel.getObservationSpace()
        self.action_space = robotModel.getActionSpace(self.isDiscreteAction)
        self.reward_range = (0.0, 1.0)
        # self.seed()
        ### pybullet settings
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotModel = robotModel
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
        self.robot = self.robotModel(isDiscreteAction=self.isDiscreteAction)
        self.foodIds = []
        for foodPos in self._generateFoodPositions(self.NUM_FOODS):
            foodId = p.loadURDF('sphere2red.urdf', foodPos, globalScaling=1.0)
            self.foodIds.append(foodId)
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
        obs = self.robot.getObservation()
        return obs

    def step(self, action):
        self.steps += 1
        self.robot.setAction(action)
        reward = 0
        for i in range(self.BULLET_STEPS_PER_GYM_STEP):
            p.stepSimulation()
            reward += self._getReward()
        self.episode_rewards += reward
        obs = self.robot.getObservation()
        done = self.steps >= self.MAX_STEPS or len(self.foodIds) <= 0
        robotPos, robotOrn = self.robot.getPositionAndOrientation()
        info = { 'steps': self.steps, 'pos': robotPos, 'orn': robotOrn }
        if done:
            info['episode'] = { 'r': self.episode_rewards, 'l': self.steps }
            # print(self.episode_rewards, self.steps)
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        _, seed = np.utils.seeding.np_random(seed)
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
