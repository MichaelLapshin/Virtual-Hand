"""
[ModelTrainerV2.py]
@description: Script for that is called by the C# script to train the model that will control the hand.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warnings (temporary)

import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm
import h5py
import math
import time
import virtualenv

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from typing import Any, List, Sequence, Tuple

# Reads the arguments
dataset_name = input()
model_name = input()

# Obtains data
data_set = h5py.File("C:\\Git\\Virtual-Hand\\PythonScripts\\training_datasets\\" + dataset_name + ".hdf5", 'r')
DATA_PER_LIMB = 2  # todo, set step to 3 when you introduce acceleration
number_of_sensors = len(data_set.get("sensor"))
number_of_limbs = len(data_set.get("angle")) * len(data_set.get("angle")[0])
possible_forces = [-10, 10]

eps = np.finfo(np.float32).eps.item()  # todo, maybe this'll mess something up


# Actor definition
class CustomModel(keras.Model):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    def __init__(self, env_name: str, num_inputs: int, num_hidden: int, num_actions: int,
                 seed=43, gamma=0.90):
        super().__init__()

        self.env = gym.make(env_name)

        # Training environment definition
        self.seed = seed
        self.gamma = gamma  # Discount for past rewards
        self.env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.num_actions = num_actions

        # todo, note, layers.Input is a Keras symbolic thingy
        # inputs = layers.Input(shape=(num_inputs,))
        inputs = layers.InputLayer(input_shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        actions = layers.Dense(num_actions, activation="softmax")(common)
        critics = layers.Dense(1)(common)

        # Creates model layers
        self.inputs = inputs
        self.outputs = [actions, critics]

        # Create gradient tape for self
        self.tape = tf.GradientTape()
        # self.tape = tf.GradientTape(persistent=True)
        # self.tape = tf.GradientTape(persistent=False)
        self.state = self.env.reset()

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def predict_torques(self, current_inputs: np.ndarray, angle_reward):
        self.state = tf.convert_to_tensor(self.state)
        self.state = tf.expand_dims(self.state, 0)

        # Predict action probabilities
        action_probs, critic_value = self(self.state)
        self.critic_prob_history.append(critic_value[0, 0])  # todo, not sure what [0,0] is

        # Sample action from action probability distribution
        action = np.random.choice(self.num_actions,
                                  p=np.squeeze(action_probs))  # the index for a list of actions to take
        self.action_probs_history.append(tf.math.log(action_probs[0, action]))

        self.state, reward, done, _ = self.env.step(action)
        self.rewards_history.append(reward)
        self.rewards_history.append(angle_reward)

        return possible_forces[action]

    def reward_fnc(self, unity_angle, expected_angle):
        return tf.constant(ANGLE_THRESHOLD * ANGLE_THRESHOLD / 1.5) - tf.square(expected_angle - unity_angle)

    def correct_model(self):
        # Update running reward to check condition for solving
        # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in self.rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(self.action_probs_history, self.critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        # loss_value = sum(actor_losses) + sum(critic_losses)
        loss_value = tf.cast(sum(actor_losses) + sum(critic_losses), tf.float32) # todo, iddue here?

        with tf.GradientTape() as tape:
            grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Clear the loss and reward history
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        self.rewards_history.clear()

    # TODO, finish this

    def reset_env(self):
        self.state = self.env.reset()


# Creates the models
models = []
number_of_inputs = number_of_sensors + number_of_limbs * DATA_PER_LIMB
for i in range(0, number_of_limbs):
    models.append(CustomModel(env_name="CartPole-v0",
    # models.append(CustomModel(env_name="HandManipulateBlock-v0",
                              num_inputs=number_of_inputs,
                              num_hidden=number_of_limbs * DATA_PER_LIMB,
                              num_actions=len(possible_forces)))
    # models[-1].build(input_shape=(number_of_inputs,))
    models[-1].build(input_shape=(35,))

# Compiles the models todo, not necessary?
# for model in models:
#     model = model.compile()

time.sleep(0.5)  # todo, to remove?

"""
    Exchanges data between data reference file and Unity
"""
print("Ready")  # Tells unity to start the training sequence
string_starting_angles = ""
for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        string_starting_angles += " " + str(data_set.get("angle")[finger_index][limb_index][0])
string_starting_angles = string_starting_angles.lstrip(" ")
print(string_starting_angles)

input_train_sensors = data_set.get("sensor")
input_train_frame_time = data_set.get("time")
input_train_angles = data_set.get("angle")

"""
    Training starts below
"""
ANGLE_THRESHOLD = 10 / 180.0 * math.pi  # +- 10 degrees from actual angle
TOTAL_NUMBER_OF_FRAMES = data_set.get("time").len()
current_frame_number = 0

while (current_frame_number != TOTAL_NUMBER_OF_FRAMES):
    unity_frame_angles = []
    expected_frame_angles = []
    current_frame_number = 0
    failed_episode = False
    for model in models:
        model.reset_env()

    while not failed_episode:
        is_unity_ready = input()
        if is_unity_ready != "Ready":
            print(
                "ERROR. C# Unity Script sent improper command. 'Ready' not receives. Received " + is_unity_ready + " instead.")

        print(data_set.get("time")[current_frame_number])

        # Receives frame capture time from unity
        unity_frame_time = int(input())

        # Obtains limb data from the C# Unity script
        string_limb_data = input().rstrip(" \n\r").split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(float(string_limb_data[i]))

        # Obtains the expected output
        expected_limb_angles = []
        for finger_index in range(0, 5):
            for limb_index in range(0, 3):
                expected_limb_angles.append(data_set.get("angle")[finger_index][limb_index][current_frame_number])

        # Extracts angles from unity model for comparison
        limb_data_angles = limb_data[::DATA_PER_LIMB]

        # Assert that expected and limb data is of the same length
        assert len(limb_data_angles) == len(expected_limb_angles)

        # Time condition
        passed_time_condition = False
        if input_train_frame_time[current_frame_number] < unity_frame_time < input_train_frame_time[
            current_frame_number + 1]:
            passed_time_condition = True

        # Angle condition
        passed_angle_condition = True
        for i in range(0, len(limb_data)):
            if expected_limb_angles[i] - ANGLE_THRESHOLD > limb_data_angles[i] \
                    or limb_data_angles[i] > expected_limb_angles[i] + ANGLE_THRESHOLD:
                passed_angle_condition = False
                break

        # Loop-3 condition
        if current_frame_number >= TOTAL_NUMBER_OF_FRAMES:
            print("Quit")
            break
        elif not passed_time_condition or not passed_angle_condition:
            print("Reset")
            failed_episode = True
            break
        else:
            print("Next")
            # todo, add more info

            """
                Have access to:
                    - limb_data_angles: unity angles
                    - expected_limb_angles: real angles
            """
            expected_frame_angles.append(expected_limb_angles)  # todo, remove?
            unity_frame_angles.append(limb_data_angles)  # todo, remove?

            # Preparing sensors inputs (obtains the sensor readings for the current frame)
            current_sensor_readings = [input_train_sensors[i][current_frame_number]
                                       for i in range(0, number_of_sensors)]

            # TODO, re-write this
            current_model_inputs = np.array(limb_data + current_sensor_readings)

            next_torques = []
            for m in range(0, len(models)):
                model = models[m]
                # next_torques.append(model.predict(current_model_inputs))
                current_reward = model.reward_fnc(limb_data_angles[m], expected_limb_angles[m])
                next_torques.append(model.predict_torques(current_model_inputs, current_reward))
                # next_torques.append(model(current_model_inputs))
                # todo, might be a problem here https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient/56917148#56917148

            # Prepared the torques to send to the unity script
            string_torques = ""
            for i in range(0, number_of_limbs):
                string_torques += next_torques[i] + " "
            string_torques = string_torques.rstrip(" ")

            # Sends the torques to the unity script
            print(string_torques)

            # Increments the frame number for the next loop
            current_frame_number += 1

    if not failed_episode:
        for m in range(0, len(models)):
            models[m].save("/models/" + model_name + "_index" + m + "_CompletedTraining")
        # model.save("/models/" + model_name + "_CompletedTraining")
        break
    else:
        # TODO, apply reward from the episode here

        for model in models:  # todo, not sure if this works, maybe remove
            model.correct_model()
            # model.reset_env()

        for m in range(0, len(models)):
            models[m].save("/models/" + model_name + "_index" + str(m) + "_frame" + str(current_frame_number))
