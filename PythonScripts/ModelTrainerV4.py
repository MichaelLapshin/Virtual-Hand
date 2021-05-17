"""
[ModelTrainerV3.py]
@description: Script for that is called by the C# script to train the model that will control the hand.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warnings (temporary)

import numpy as np
import tensorflow as tf
import h5py
import math
import time
import sys

from tensorflow.keras import layers
from tensorflow import keras

from ClientConnectionHandlerV2 import ClientConnectionHandler

sys.stderr = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientError_ModelTrainer.txt", "w")
f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientLog.txt", 'w')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "\n")
f.close()
sys.stdout = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientLog.txt", 'a')

print_to_logs = False
print_critical = True


def controlled_print(controlled_message):
    """
    A print statement that can be turned off with single variable "print_to_logs".
    Used to include/remove less important information.
    """
    if print_to_logs:
        print(str(controlled_message))


def critical_print(critical_message):
    """
    A print statement that can be turned off with single variable "print_critical".
    Used to include/remove important information.
    """
    if print_critical:
        print("=== Critical: " + str(critical_message))
    else:
        print_to_logs(critical_message)


connection_handler = ClientConnectionHandler()

# Reads the arguments
dataset_name = connection_handler.input()
model_name = connection_handler.input()
critical_print("Received first C# inputs: " + dataset_name + " " + model_name)

# Obtains data
data_set = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + dataset_name + ".hdf5", 'r')
DATA_FRAMES_PER_SECOND = 50
DATA_PER_LIMB = 2  # todo, set step to 3 when you introduce acceleration
ms_time_per_data_frame = 1000 / DATA_FRAMES_PER_SECOND
NUMBER_OF_SENSORS = len(data_set.get("sensor"))
NUMBER_OF_LIMBS = len(data_set.get("angle")) * len(data_set.get("angle")[0])
NUMBER_OF_INPUTS = NUMBER_OF_SENSORS + NUMBER_OF_LIMBS * DATA_PER_LIMB

ANGLE_THRESHOLD_RADIANS = 3 * math.pi / 180.0  # +- 10 degrees from actual angle threshold // TODO, modify this constant as needed
# MAX_REWARD_ANGLE_DIF_RADIANS = 3 * math.pi / 180.0  # the distance off-perfect which is given max reward

# possible_forces = [-0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002, -0.0001,
#                    0.0,
#                    0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128]
# possible_forces = [0, -0.035, 0.035, -0.075, 0.075, -0.125, 0.125, -0.25, 0.25, 0.5, 0.5, -1, 1, -2, 2, -4, 4]
# possible_forces = [-20, 0.0]
MIN_MAX_OUTPUT = 100000
ACTION_DIVISOR = 1000.0

# For a given limb, the following are constants that contribute to its reward
REWARD_MAX_GAIN = 10
REWARD_HIGH_GAIN = 0
REWARD_OTHER_GAIN = 0
REWARD_EXPONENT = 2
LEARNING_RATE = 0.01
NUMBER_OF_DENSE_NEURONS = 4  # TODO, switch this back at some point to non-zero
# DENSE_ACTIVATION_FUNCTIONS = ["linear", "linear"]
DENSE_ACTIVATION_FUNCTIONS = []

# Constants to determine which frames are to be used
EARLIEST_FRAME_TO_USE = 1
MAX_CONSECUTIVE_FAILED_FRAMES = 1
failed_frame_count = 0

eps = np.finfo(np.float32).eps.item()


# Actor definition
class CustomModel:

    def __init__(self, gamma=0.99):
        # Object variables (non-static variables)
        # self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        # self.optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        self.optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
        self.huber_loss = keras.losses.Huber()
        self.action_history = []
        # self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.current_action_index = None
        self.NUM_ACTIONS = 1  # todo, hard code this?

        # Normal constructor stuff
        self.gamma = gamma

        # self.NUM_ACTIONS = len(possible_forces)

        # Creates the model for the agent
        inputs = layers.Input(shape=(NUMBER_OF_INPUTS,))  # todo, change back?
        common1 = layers.Dense(NUMBER_OF_DENSE_NEURONS, activation="linear",
                               kernel_initializer=keras.initializers.RandomNormal,
                               bias_initializer=keras.initializers.RandomNormal(mean=0, stddev=1)
                               )(inputs)
        for f in DENSE_ACTIVATION_FUNCTIONS:
            common2 = layers.Dense(NUMBER_OF_DENSE_NEURONS, activation=f,
                                   kernel_initializer=keras.initializers.zeros,
                                   bias_initializer=keras.initializers.Zeros()
                                   )(common1)
            common1 = common2
        action = layers.Dense(1, activation="linear",
                              kernel_initializer=keras.initializers.RandomNormal,
                              bias_initializer=keras.initializers.RandomNormal(mean=0, stddev=1)
                              )(common1)
        # )(dense)
        self.model = keras.Model(inputs=inputs, outputs=action)

        self.tape = tf.GradientTape()

    def step(self, state: tuple, reward: float):
        with self.tape as tape:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards from environment state
            action = self.model(state)[0]
            controlled_print("Action: " + str(action))
            # critical_print("Critic value: " + str(critic_value)) # todo, remove this later
            # self.critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            # action = np.random.choice(self.NUM_ACTIONS, p=np.squeeze(action_probs))
            # self.action_probs_history.append(tf.math.log(action_probs[0, action]))

            self.action_history.append(action)
            self.rewards_history.append(reward)

        action = min(max(action, np.float(-MIN_MAX_OUTPUT)), np.float(MIN_MAX_OUTPUT)) / ACTION_DIVISOR
        critical_print("ACTION ACTION ACTION: " + str(action))
        return action

    def backpropagate_model(self):
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic

        if EARLIEST_FRAME_TO_USE >= len(self.action_history):
            return

        with self.tape as tape:
            # cropped_action_history = self.action_history[EARLIEST_FRAME_TO_USE::]  # Todo, remove?
            # cropped_rewards_history = self.rewards_history[EARLIEST_FRAME_TO_USE::]
            # cropped_critic_value_history = self.critic_value_history[EARLIEST_FRAME_TO_USE::]

            cropped_action_history = [self.action_history[-1]]  # Todo, remove?
            cropped_rewards_history = [self.rewards_history[-1]]

            returns = []
            discounted_sum = 0
            # for r in self.rewards_history[1::-1]:  # TODO, fix here?
            # p = 0
            for r in cropped_rewards_history[::-1]:  # TODO, fix here?
                discounted_sum = r + self.gamma * discounted_sum

                # discounted_sum = r + p * self.gamma
                # p = r
                returns.insert(0, discounted_sum)
                if r == cropped_action_history[-1]:  # TODO, remove this statement?
                    discounted_sum = 0
            critical_print("self.reward_history length: " + str(len(cropped_action_history)))
            critical_print("DISCOUNTED SUM: " + str(discounted_sum))

            # Normalize todo, problem?
            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(cropped_action_history, returns)
            actor_losses = []
            # critic_losses = []
            # for log_prob, value, ret in history:
            for act, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret  # - value
                # actor_losses.append(-act * diff)  # actor loss
                actor_losses.append(self.huber_loss(tf.expand_dims(act, 0), tf.expand_dims(ret, 0)))  # actor loss
                # actor_losses.append(act * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of the future rewards.
                # critic_losses.append(
                #     self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                # )
                critical_print("ACTOR LOSSES: " + str(actor_losses))

            # Backpropagation
            loss_value = tf.constant(sum(actor_losses), dtype=tf.float32)
            tape.watch(self.model.trainable_variables)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            critical_print("Gradients === " + str(grads))

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def r2d(self, radians):  # radian to degrees conversion
        return radians * 180 / math.pi

    def individual_angle_reward(self, unity_angle, expected_angle, unity_velocity, expected_velocity):
        # This reward gives a "flat" reward when the difference approaches 0,
        # but a very negative beyond the ANGLE_THRESHOLD_RADIANS

        # if math.fabs(expected_angle - unity_angle) <= MAX_REWARD_ANGLE_DIF_RADIANS:
        #     return math.pow(self.r2d(ANGLE_THRESHOLD_RADIANS), 2)
        # else:
        #     return max(-math.pow(self.r2d(ANGLE_THRESHOLD_RADIANS), 2),
        #                 -math.pow(self.r2d(expected_angle - unity_angle), 2)
        #                 + math.pow(self.r2d(ANGLE_THRESHOLD_RADIANS), 2))
        # todo, GRADIENT DESCENT!
        return -(-math.pow(self.r2d(math.fabs(expected_angle - unity_angle)), REWARD_EXPONENT) + math.pow(
            self.r2d(ANGLE_THRESHOLD_RADIANS), REWARD_EXPONENT))
        # return -(-math.pow(self.r2d(math.fabs(unity_velocity - expected_velocity))*2,REWARD_EXPONENT)-math.pow(self.r2d(math.fabs(expected_angle - unity_angle))*2,REWARD_EXPONENT) + math.pow(self.r2d(ANGLE_THRESHOLD_RADIANS), REWARD_EXPONENT))
        # return max(0.0,
        #            -math.pow(self.r2d(expected_angle - unity_angle), 2)
        #            + math.pow(self.r2d(ANGLE_THRESHOLD_RADIANS), 2))

    def compute_reward(self, unity_angles, expected_angles, unity_velocities, expected_velocities,
                       max_reward_indices,
                       high_reward_indices=[],
                       other_reward_indices=[]):

        # Computes the reward for a limb based on all of the listed related limbs
        limb_reward = 0.0
        for r in max_reward_indices:
            limb_reward += self.individual_angle_reward(unity_angles[r], expected_angles[r], unity_velocities[r],
                                                        expected_velocities[r]) * REWARD_MAX_GAIN
        for r in high_reward_indices:
            limb_reward += self.individual_angle_reward(unity_angles[r], expected_angles[r], unity_velocities[r],
                                                        expected_velocities[r]) * REWARD_HIGH_GAIN
        for r in other_reward_indices:
            limb_reward += self.individual_angle_reward(unity_angles[r], expected_angles[r], unity_velocities[r],
                                                        expected_velocities[r]) * REWARD_OTHER_GAIN

        return limb_reward

    def reset_episode(self):
        with self.tape as tape:
            # Clear the loss and reward history
            self.action_history.clear()
            # self.critic_value_history.clear()
            self.rewards_history.clear()

    def save(self, file_name: str):
        self.model.save(file_name)


# Creates the models
models = []
for i in range(0, NUMBER_OF_LIMBS):
    models.append(CustomModel(gamma=0.90))

"""
    Exchanges data between data reference file and Unity
"""
controlled_print("Sending Ready to C#")
connection_handler.print("Ready")  # Tells unity to start the training sequence

temp = connection_handler.input()
if temp != "Ready":
    connection_handler.print("ERROR. Did not receive ready from C#. Received: " + temp)

string_starting_angles = ""
starting_angles = []
for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        string_starting_angles += " " + str(data_set.get("angle")[finger_index][limb_index][0])
        starting_angles.append(data_set.get("angle")[finger_index][limb_index][0])
string_starting_angles = string_starting_angles.lstrip(" ")

controlled_print("Sending starting angles to C#")
connection_handler.print(string_starting_angles)

input_train_sensors = data_set.get("sensor")
input_train_frame_time = data_set.get("time")
input_train_angles = data_set.get("angle")

"""
    Training starts below
"""
TOTAL_NUMBER_OF_FRAMES = data_set.get("time").len()
# current_frame_number = 0
latest_unity_time = 0

# while current_frame_number != TOTAL_NUMBER_OF_FRAMES:
while latest_unity_time < TOTAL_NUMBER_OF_FRAMES * ms_time_per_data_frame:
    # current_frame_number = 0
    latest_unity_time = 0

    failed_episode = False
    for model in models:
        model.reset_episode()

    connection_handler.print("Ready")
    failed_frame_count = 0
    next_torques = [0 for i in range(0, NUMBER_OF_LIMBS)]
    previous_state = None
    # rewards = [0 for k in range(0, 15)]

    while not failed_episode:
        is_unity_ready = connection_handler.input()
        if is_unity_ready != "Ready":
            connection_handler.print(
                "ERROR. C# Unity Script sent improper command. 'Ready' not received. Received " + is_unity_ready + " instead.")

        # Receives frame capture time from unity
        latest_unity_time = int(connection_handler.input())
        latest_frame_index = int(latest_unity_time / ms_time_per_data_frame)

        # connection_handler.print(data_set.get("time")[current_frame_number]) # todo, removed because reactive model
        critical_print("unity time: " + str(latest_unity_time) + "    index conversion: " + str(latest_frame_index))

        # Obtains limb data from the C# Unity script
        string_limb_data = connection_handler.input().rstrip(" $").split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(float(string_limb_data[i]))

        # Obtains the expected output
        expected_limb_angles = []
        expected_limb_velocities = []
        for finger_index in range(0, 5):
            for limb_index in range(0, 3):
                expected_limb_angles.append(data_set.get("angle")[finger_index][limb_index][latest_frame_index])
                expected_limb_velocities.append(data_set.get("velocity")[finger_index][limb_index][latest_frame_index])

        # Sends the expected angles to the C# script for display
        stringExpectedAngles = ""
        for a in expected_limb_angles:
            stringExpectedAngles += " " + str(a)
        stringExpectedAngles = stringExpectedAngles.rstrip(' ')

        connection_handler.print(stringExpectedAngles)

        # Extracts angles from unity model for comparison
        current_limb_angles = limb_data[::DATA_PER_LIMB]
        current_limb_velocities = limb_data[1::DATA_PER_LIMB]

        # Assert that expected and limb data is of the same length
        assert len(current_limb_angles) == len(expected_limb_angles)

        # Time condition // todo, no more time condition because of reactive approach
        # passed_time_condition = False
        # if input_train_frame_time[current_frame_number] <= unity_frame_time <= input_train_frame_time[
        #     current_frame_number + 1]:
        #     passed_time_condition = True
        # passed_time_condition = True  # todo, mess around with this

        # Angle condition
        passed_angle_condition = True
        why_failed = 0
        for i in range(0, len(expected_limb_angles)):
            if expected_limb_angles[i] - ANGLE_THRESHOLD_RADIANS > current_limb_angles[i] \
                    or current_limb_angles[i] > expected_limb_angles[i] + ANGLE_THRESHOLD_RADIANS:
                passed_angle_condition = False
                why_failed = i
                # break # TODO, revert back?
        # passed_angle_condition = True # TODO, REMOVE THIS WHEN DONE DEBUGGING
        if passed_angle_condition == False:
            failed_frame_count += 1
        else:
            failed_frame_count = 0

        # Preparing sensors inputs (obtains the sensor readings for the current frame)
        current_sensor_readings = [input_train_sensors[i][latest_frame_index] for i in range(0, NUMBER_OF_SENSORS)]

        # Preparing the model inputs (Gathers the data + converts to tuple)
        # previous_state = current_state  # todo, new variable used to "shift" the reward-state assignment
        current_state = (np.array(limb_data + current_sensor_readings))  # + next_torques))

        next_torques = []
        for m in range(0, len(models)):
            model = models[m]

            max_list = [m]
            high_list = []
            # high_list = [int(m / 3) * 3, int(m / 3) * 3 + 1, int(m / 3) * 3 + 2]
            # high_list.remove(m)

            if previous_state is None:
                previous_state = [0 for k in range(0, NUMBER_OF_INPUTS)]
                reward = model.compute_reward(starting_angles, starting_angles, starting_angles, starting_angles,
                                              max_list, high_list, [])
            else:
                reward = model.compute_reward(current_limb_angles, expected_limb_angles, current_limb_velocities,
                                              expected_limb_velocities, max_list, high_list,
                                              [])  # TODO, add more dependencies (or less)
            # reward = 0
            # if failed_episode:
            #     reward = 0
            # else:
            #     reward = 1

            # next_torques.append(possible_forces[model.step(state=previous_state, reward=reward)])
            next_torques.append(float(model.step(state=previous_state, reward=reward)))
            # a = [0 for k in range(0, int(m / 3) * 3)]
            # b = list(previous_state[int(m / 3) * 3:int(m / 3) * 3 + 3:])
            # c = [0 for k in range(int(m / 3) * 3 + 3, 30)]
            # d = list(previous_state[30::])
            # e = a + b + c + d
            # next_torques.append(possible_forces[model.step(state=e, reward=reward)])

            # next_torques.append(possible_forces[model.step(state=current_state, reward=reward)])

            # next_torques.append(possible_forces[model.step(state=previous_state, reward=reward)])
            controlled_print("REWARD: " + str(reward))

        # rewards = [0 for k in range(0,15)]
        previous_state = current_state

        # Loop-3 condition
        if latest_unity_time >= TOTAL_NUMBER_OF_FRAMES * ms_time_per_data_frame:
            # if current_frame_number >= TOTAL_NUMBER_OF_FRAMES:
            connection_handler.print("Quit")
            break
        elif failed_frame_count >= MAX_CONSECUTIVE_FAILED_FRAMES:
            # elif not passed_time_condition or not passed_angle_condition:
            connection_handler.print("Reset")
            failed_episode = True
            critical_print("Triggering reset... Did not pass time condition.")
            # if not passed_time_condition:
            #     critical_print("Did not pass time condition.")
            # if not passed_angle_condition:

            critical_print("Did not pass angle condition. Index: " + str(why_failed) + "  Expected: " + str(
                expected_limb_angles[why_failed]) + "  Got: " + str(
                current_limb_angles[why_failed]) + "  Difference:" + str(
                expected_limb_angles[why_failed] - current_limb_angles[why_failed]))
            break
        else:
            connection_handler.print("Next")
            """
                Have access to:
                    - current_limb_angles: unity angles
                    - expected_limb_angles: real angles
            """
            critical_print("Next is initiated. Frame: " + str(latest_frame_index))

            # Prepared the torques to send to the unity script
            string_torques = ""
            for i in range(0, NUMBER_OF_LIMBS):
                string_torques += str(next_torques[i]) + " "
            string_torques = string_torques.rstrip(" ")

            # Sends the torques to the unity script
            connection_handler.print(string_torques)

            # Increments the frame number for the next loop
            # current_frame_number += 1

    if not failed_episode:
        for m in range(0, len(models)):
            models[m].save("/models/" + model_name + "_index" + m + "_CompletedTraining")
        break
    else:

        for model in models:
            model.backpropagate_model()  # TODO, don't forget to include this

        # if current_frame_number % 50 == 0:  # Saves temporary models every 30 frames (if failed on multiple of 30 frames)
        # for m in range(0, len(models)):
        # models[m].save("/models/" + model_name + "_index" + str(m) + "_frame" + str(current_frame_number))
        # pass

critical_print("EVERYTHING IS DONE! CONGRATS (or not if its actually broken)!")
