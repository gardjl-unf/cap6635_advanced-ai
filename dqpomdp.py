#!/usr/bin/env python3

__author__ = "Jason Gardner"
__credits__ = ["Jason Gardner"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jason Gardner"
__email__ = "n01480000@unf.edu"
__status__ = "Production"
__references__ = ["[1] Hausknecht, Matthew J. and Peter Stone. “Deep Recurrent Q-Learning for Partially Observable MDPs.” ArXiv abs/1507.06527 (2015): n. pag.",
                  "[2] Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, Graves A, Riedmiller M, Fidjeland AK, Ostrovski G, Petersen S, Beattie C, Sadik A, Antonoglou I, King H, Kumaran D, Wierstra D, Legg S, Hassabis D. Human-level control through deep reinforcement learning. Nature. 2015 Feb 26;518(7540):529-33. doi: 10.1038/nature14236. PMID: 25719670."]

import argparse
import logging
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import uuid
import time
import pickle
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM, Reshape
from collections import deque
from gym.wrappers.atari_preprocessing import AtariPreprocessing as atari_preprocessing
from gym.wrappers.transform_reward import TransformReward as transform_reward
import numpy as np
import gym
import tensorflow as tf

TEST = False
DEBUG = False

'''
TODO: Load model from UUID?  Play game for a set amount of time using that model, or continue training it.
TODO: Clean up test/debug code using flags

'''

# Disable TensorFlow logging
tf.keras.utils.disable_interactive_logging()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

INPUT_DIMENSION = 84
LOG_FORMAT_STRING = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")
#RMSPROP_CLIP = 10.0
# The following ALE options were used: color averaging, minimal action set, and death detection. [1]

'''
First, experiences et = (st,at,rt,st+1) are recorded in a replay memory D and then sampled uniformly at training time. 
Second, a separate, target network ˆQ provides update targets to the main network, decoupling the feedback resulting 
from the network generating its own targets. ˆQ is identical to the main network except its parameters θ− are updated 
to match θ every 10,000 iterations. Finally, an adaptive learning rate method such as RMSProp (Tieleman and Hinton 2012) 
or ADADELTA (Zeiler 2012) maintains a per-parameter learning rate α, and adjusts α according to the history of gradient
updates to that parameter. [1]
'''

###############
###  MODEL  ###
###############

class Model():
   def __init__(self, actions: int = None, state : dict = None) -> None:
         self.actions = actions
         self.state = state
         self.uuid = self.state['uuid']
         self.actions = self.state['actions']
         self.model = None

         if self.state['uuid'] is not None:
            logger.info(f"Loading model from {self.state['uuid']}")
            self.init_model()
            self.load_model()
         else:
            logger.info("No model UUID provided. Generating new UUID")
            self.state['uuid'] = str(uuid.uuid4())
            self.uuid = self.state['uuid']
            logger.info(f"New UUID: {self.uuid}")
            logger.info(f"Initializing new model")
            self.init_model()

   def init_model(self) -> tf.keras.Model:
      if self.state["network"] == "DQN":
         logger.info("Initializing DQN model")
         self.model = Sequential([
                        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Input
                        # Input: Input layer for the network
                        # shape: A shape tuple (integers), not including the batch size
                        Input(shape = (INPUT_DIMENSION, INPUT_DIMENSION, 1)),
                        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
                        # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
                        # kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
                        # strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
                        # activation: Activation function to use
                        Conv2D(filters = 32, kernel_size = (8, 8), strides = 4, activation = "relu"),
                        Conv2D(filters = 64, kernel_size = (4, 4), strides = 2, activation = "relu"),
                        Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu"),
                        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
                        # Flatten: Flattens the input. Does not affect the batch size.
                        Flatten(),
                        Dense(units = 256, activation = "relu"),
                        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
                        # units: Positive integer, dimensionality of the output space
                        Dense(units = self.actions)
                        ])
      elif self.state["network"] == "DRQN":
         logger.info("Initializing DRQN model")
         self.model = Sequential([Input(shape = (INPUT_DIMENSION, INPUT_DIMENSION, 1)),
                                  Conv2D(filters = 32, kernel_size = (8, 8), strides = 4, activation = "relu"),
                                  Conv2D(filters = 64, kernel_size = (4, 4), strides = 2, activation = "relu"),
                                  Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, activation = "relu"),
                                 # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
                                 # Reshape: Reshapes an output to a certain shape
                                 # target_shape: Target shape. Tuple of integers. Does not include the batch axis.
                                 # This layer changes the shape for the LSTM layer
                                  Reshape(target_shape = (1, 3136)),
                                 # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
                                 # units: Positive integer, dimensionality of the output space
                                  LSTM(units = 512),
                                  Dense(units = self.actions)])
      else:
         logger.error("Invalid network type. Please use 'DQN' or 'DRQN'")
         raise ValueError("Invalid network type. Please use 'DQN' or 'DRQN'")
      
      #self.optimizer = tf.keras.optimizers.RMSprop(clipvalue = RMSPROP_CLIP)
      # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
      # Additionally, all networks used ADADELTA (Zeiler 2012) optimizer with a learning rate of 0.1 and momentum of 0.95 [1]
      self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
      self.loss_function = tf.keras.losses.Huber()
      # self.loss_function = tf.keras.losses.mean_squared_error
      self.model.compile(optimizer = self.optimizer, loss = self.loss_function)
      logger.info("Model initialized")
           
   def save_model(self, state: dict = None) -> None:
      if state is not None:
         self.state = state
      logger.info(f"Saving model to './models/{self.uuid}/model.h5'")
      logger.info(f"Creating directory './models/{self.state['uuid']}'")
      results_dir_path = f"./models/{self.state['uuid']}"
      logger.info(f"Saving data to './models/{self.uuid}/'")
      if not os.path.exists(results_dir_path):
         if not os.path.exists('./models'):
               try:
                  os.mkdir('./models')
               except OSError:
                  logger.warning(f"Creation of the directory {'./models'} failed")
                  exit(1)
         try:
               os.mkdir(results_dir_path)
         except OSError:
               logger.warning(f"Creation of the directory {results_dir_path} failed")
               exit(1)
         else:
               logger.info(f"Successfully created the directory {results_dir_path}")

      self.model.save_weights(f'./models/{self.state["uuid"]}/model.weights.h5')
      logger.info(f"Saved model weights to ./models/{self.state['uuid']}/model.weights.h5")

      logger.info(f"Saving Numpy random state to ./models/{self.state['uuid']}/numpy_random_state.pkl")
      with open(f"./models/{self.state['uuid']}/numpy_random_state.pkl", 'wb') as f:
         pickle.dump(np.random.get_state(), f)
      logger.info(f"Saved Numpy random state to ./models/{self.state['uuid']}/random_state.npy")

      with open(f'./models/{self.state["uuid"]}/state.json', 'w') as f:
         logger.info(f"Saving {self.state['network']} ({self.state['uuid']}) network environment.\n Game: {self.state['environment']}\n\
                     ε = {self.state['epsilon']}\n\
                     ε Step = {self.state['epsilonstep']}\n\
                     Action Space Size = {self.state['actions']}\n\
                     Initial Timestep = {self.state['interval']}\n\
                     Maximum Timesteps = {self.state['timestep']}\n\
                     Buffer Length = {self.state['buffer']}\n\
                     Evaluation Interval = {self.state['numsteps']}\n\
                     Timestep = {self.state['timestep']}\n\
                     Seed = {self.state['seed']}\n\
                     Clone Steps = {self.state['clonesteps']}\n\
                     Batch Size = {self.state['batchsize']}\n\
                     Gamma = {self.state['gamma']}\n\
                     X (# of Frames)= {self.state['x']}\n\
                     Y (Reward Value)= {self.state['y']}\n\
                     ")
         json.dump(self.state, f)
         logger.info(f"Saved state to './models/{self.uuid}/state.json'")

   def load_model(self) -> tf.keras.Model:
      logger.info(f"Loading model from './models/{self.uuid}/model.weights.h5'")
      self.model.load_weights(f'./models/{self.uuid}/model.weights.h5')
      logger.info(f"Loaded model weights from './models/{self.uuid}/model.weights.h5'")
      
      with open(f"./models/{self.state['uuid']}/numpy_random_state.pkl", 'rb') as f:
         random_state = pickle.load(f)
         np.random.set_state(random_state)
      logger.info(f"Loaded Numpy random state from './models/{self.uuid}/numpy_random_state.pkl'")

      if os.path.exists(f'./models/{self.uuid}/state.json'):
         with open(f'./models/{self.uuid}/state.json', 'r') as f:
            logger.info(f"Loading state from './models/{self.uuid}/state.json'")
            self.state = json.load(f)
            logger.info(f"Loading {self.state['network']} ({self.state['uuid']}) network environment.\n Game: {self.state['environment']}\n\
                     ε = {self.state['epsilon']}\n\
                     ε Step = {self.state['epsilonstep']}\n\
                     Action Space Size = {self.state['actions']}\n\
                     Initial Timestep = {self.state['interval']}\n\
                     Maximum Timesteps = {self.state['timestep']}\n\
                     Buffer Length = {self.state['buffer']}\n\
                     Evaluation Interval = {self.state['numsteps']}\n\
                     Timestep = {self.state['timestep']}\n\
                     Seed = {self.state['seed']}\n\
                     Clone Steps = {self.state['clonesteps']}\n\
                     Batch Size = {self.state['batchsize']}\n\
                     Gamma = {self.state['gamma']}\n\
                     X = {self.state['x']}\n\
                     Y = {self.state['y']}\n\
                     ")
            logger.info(f"Loaded state from './models/{self.uuid}/state.json'")     

      return self.model

###############
### LOGGING ###
###############
         
class Logging:
   def __init__(self, logger_name: str = '__main__') -> None:
      self.logger = logging.getLogger(logger_name)
      self.logger.setLevel(logging.DEBUG)
      self.logger.addHandler(self.get_console_handler())
      self.logger.propagate = False
      self.logger.info(f"Logging initialized -- {logger_name}")

   def get_console_handler(self) -> logging.StreamHandler:
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(LOG_FORMAT_STRING)

      return console_handler

   def get_logger(self) -> logging.Logger:

      return self.logger
   
#################
###  PARSER   ###
#################

class Arguments(argparse.ArgumentParser):
   def __init__(self) -> None:
      '''
      Policies were evaluated every 50,000 iterations by playing
      10 episodes and averaging the resulting scores. Networks
      were trained for 10 million iterations and used a replay
      memory of size 400,000. [1]
      '''
      super().__init__(prog = "DQPOMDP", 
                       description = "Train a DQN or DRQN to play a game",
                       allow_abbrev = True)
      
      self.add_argument("-r", 
                        "--render", 
                        help = "render to display", 
                        default = False, 
                        action = "store_true")
      
      self.add_argument("-f", 
                        "--frameskip", 
                        help = "skip every nth frame", 
                        type = int, 
                        default = 4)
      
      self.add_argument("-q",
                        "--framesequence",
                        help = "sequence of frames to use",
                        type = int,
                        default = 4)
      
      self.add_argument("-v", 
                        "--environment", 
                        help = "environment name", 
                        type = str, 
                        default = "ALE/Centipede-v5")
      
      self.add_argument("-l", 
                        "--listenvs", 
                        help = "list valid environments", 
                        default = False, 
                        action = "store_true")
      
      self.add_argument("-n", 
                        "--network", 
                        help = "the type of network to load: DQN or DRQN", 
                        type = str, 
                        default = "DRQN")
      
      self.add_argument("-u", 
                        "--uuid", 
                        help = "the model UUID to load", 
                        type = str, 
                        default = None)
      
      self.add_argument("-p", 
                        "--epsilon", 
                        help = "the initial epsilon to use", 
                        type = float, 
                        default = 1.0)
      
      self.add_argument("-e", 
                        "--epsilonstep", 
                        help = "the initial epsilon step to use", 
                        type = float, 
                        default = 18e-7)
      
      self.add_argument("-t", 
                        "--timestep", 
                        help = "the initial timestep to use", 
                        type = int, 
                        default = 0)
      
      self.add_argument("-a",
                        "--updatefrequency",
                        help = "the frequency to update the target network",
                        type = int,
                        default = 10000)
      
      self.add_argument("-b", 
                        "--buffer", 
                        help = "the length of the buffer", 
                        type = int, 
                        default = 400000)
      
      self.add_argument("-m", 
                        "--numsteps", 
                        help = "the maximum steps to run", 
                        type = int,
                        default = 10000000)
      
      self.add_argument("-i", 
                        "--interval",
                        help = "the interval in which to evaluate", 
                        type = int, 
                        default = 50000)
      
      self.add_argument("-d",
                        "--seed",
                        help = "the seed to use",
                        type = int,
                        default = 42)
      
      self.add_argument("-c",
                        "--clonesteps",
                        help = "the number of steps to clone",
                        type = int,
                        default = 10000)

      self.add_argument("-z",
                        "--batchsize",
                        help = "the batch size",
                        type = int,
                        default = 32)
      
      self.add_argument("-g",
                        "--gamma",
                        help = "the gamma value",
                        type = float,
                        default = 0.99)
      
      self.add_argument("-T",
                        "--test",
                        help = "run test",
                        default = False,
                        action = "store_true")
      
      self.add_argument("-D",
                        "--debug",
                        help = "run debug",
                        default = False,
                        action = "store_true")
      
      self.add_argument("-y",
                        "--play",
                        help = "play the game using an existing network",
                        default = False,
                        action = "store_true")
      
###############
###  AGENT  ###
###############
      
class Agent:
   def __init__(self, state: dict) -> None:
      self.min_reward = -1
      self.max_reward = 1
      self.state = state
      self.seed = state['seed']
      self.environment = state['environment']
      self.uuid = state['uuid']
      self.epsilon = state['epsilon']
      self.epsilonStep = state['epsilonstep']
      self.time = state['timestep']
      self.max_time_steps = state['numsteps']
      self.buffer_length = state['buffer']
      self.buffer = deque(maxlen = self.buffer_length)
      self.eval = state['interval']
      self.render = state['render']
      self.frame_skip = state['frameskip']
      # The following ALE options were used: color averaging, minimal action set, and death detection. [1]
      if self.render is False:
         #self.env = GymWrapper(gym.make(state['environment']), state['frameskip'])
         self.env = gym.make(self.environment,
                             frameskip = 1, 
                             full_action_space = False, 
                             repeat_action_probability = 0.25)
      else:
         #self.env = GymWrapper(gym.make(state['environment'], render_mode='human'), state['frameskip'])
         self.env = gym.make(state['environment'], 
                             render_mode='human', 
                             frameskip = 1, 
                             full_action_space = False, 
                             repeat_action_probability = 0.25)
                             
      if self.env.unwrapped.ale.getAvailableDifficulties() is not None:
         self.env.unwrapped.ale.setDifficulty(self.env.unwrapped.ale.getAvailableDifficulties()[-1])
      self.env = atari_preprocessing(self.env,
                        frame_skip = self.frame_skip,
                        screen_size = INPUT_DIMENSION, 
                        terminal_on_life_loss = True, 
                        grayscale_obs = True,
                        scale_obs = True)
      self.env = transform_reward(self.env, lambda reward: np.clip(reward, self.min_reward, self.max_reward))
      self.env.metadata['render_fps'] = 360
      self.env.seed(self.seed)
      self.env.reset()
      state["actions"] = self.env.action_space.n
      self.num_actions = state['actions']
      self.clone_steps = state['clonesteps']
      self.batch_size = state['batchsize']
      self.frame_sequence = state['framesequence'] if state['network'] == "DRQN" else 1
      self.gamma = state['gamma']
      self.S = None
      self.X = state['x']
      self.Y = state['y']
      np.random.seed(self.seed)
      # Additionally, all networks used ADADELTA (Zeiler 2012) optimizer with a learning rate of 0.1 and momentum of 0.95 [1]
      self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.1, ema_momentum = 0.95)
      self.loss_function = tf.keras.losses.Huber()
      self.M = Model(actions = self.num_actions, state = state)
      self.M.model.summary()
      self.model = self.M.model
      self.q_model = tf.keras.models.clone_model(self.model)
      self.q_model.set_weights(self.model.get_weights())
      self.update_frequency = state['updatefrequency'] + 1
      
      self.debug = None

   def run(self):   
      self.S = self.env.reset()

      if self.state['play'] is True:
         self.play_game()

      if parser.data['test'] is True:
         self.test_populate_buffer_with_dummy_data()
         self.training_step()

         return self.model
     
      for _ in range(self.max_time_steps):
         self.time += 1
         
         if self.time != 0 and self.time % self.eval == 0 and self.time > len(self.buffer):
            self.X.append(self.time)
            self.Y.append(self.evaluate())

         if self.render:
            self.env.render()
            
         if self.time != 0 and self.time % self.clone_steps == 0:
            self.q_model.set_weights(self.model.get_weights())

         self.epsilon = max(0.1, self.epsilon - self.epsilonStep)

         action = self.get_action(self.epsilon)
         #logger.info(f"Step: {self.time}, Action: {action}, Buffer Length: {len(self.buffer)}, ε: {self.epsilon}")
         self.S, _, done = self.play_step(action)
         
         if done:
               self.S = self.env.reset()[0]
               
         if self.time > len(self.buffer):
            self.training_step()    
               
      return self.model

   def play_step(self, action):
      #obs, reward, terminated, truncated , info = self.env.step(action)
      S_tag, reward, done, _, _ = self.env.step(action)

      self.buffer.append([self.S, action, reward, S_tag, done])
         
      if DEBUG:
         experience = self.buffer[0]
         this_experience = ""
         for i, element in enumerate(experience):
            if isinstance(element, np.ndarray):
                  this_experience += str(element.shape)
            else:
                  this_experience += str(type(element))
            if i < len(experience) - 1:
                  this_experience += ", "
         if self.debug is None:
            self.debug = this_experience
         if this_experience != self.debug:
            logger.error("Malformed buffer entry:")
            logger.error(f"Expected: {self.debug}")
            logger.error(f"Received: {this_experience}")

      return S_tag, reward, done

   def get_action(self, epsilon):
      if np.random.rand() < epsilon:
         return np.random.randint(self.num_actions)
      else:
         Q = self.model.predict(self.S[np.newaxis])
         return np.argmax(Q[0])

   def training_step(self):
      experiences = self.get_experiences()
      states, actions, rewards, next_states, dones = experiences
      states = states.astype(np.float32)
      next_Q_values = self.q_model.predict(next_states)
      max_next_Q_values = np.max(next_Q_values, axis = 1)
      target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
      mask = tf.one_hot(actions, self.num_actions)
      with tf.GradientTape() as tape:
         all_Q_values = self.model(states)
         Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims = True)
         loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
         gradients = tape.gradient(loss, self.model.trainable_variables)
         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

   def get_experiences(self):
      indices = np.random.randint(len(self.buffer) - self.frame_sequence + 1, size = self.batch_size)
      if self.frame_sequence > 1:
         new_indices = []
         for i in list(indices):
            for j in range(self.frame_sequence - 1):
               new_indices.append(i + j)
               indices = new_indices
   
      sequences = [self.buffer[i] for i in indices]

      states, actions, rewards, next_states, dones = [
         np.array([experience[field_index] for experience in sequences]) for field_index in range(5)]
            
      return states, actions, rewards, next_states, dones

   def evaluate(self, num_episodes = 10, max_episode_time = 1) -> float:
      #Policies were evaluated every 50,000 iterations by playing 10 episodes and averaging the resulting scores. 
      values = []
      self.M.save_model(self.state)
      for i in range(num_episodes):
         self.env.reset()
         value = 0
         start_time = time.time()
         done = False
         while (time.time() - start_time) < max_episode_time * 60 and not done:
               if self.render:
                  self.env.render()
               action = self.get_action(epsilon = 0.0)
               S, R, done, _, _ = self.env.step(action)
               self.S = S
               value += R
         logger.info(f"Evaluating: Iteration {i} of {num_episodes}, Value: {value}")
         values.append(value)
      return np.mean(values)
   
   def test_populate_buffer_with_dummy_data(self):
      for _ in range(500):
         S = np.random.random((84, 84, 1))
         A = np.random.randint(0, 18)
         R = np.random.random()
         S_tag = np.random.random((84, 84, 1))
         done = False
         self.buffer.append([S, A, R, S_tag, done])
         
   def play_game(self):
      while True:
         done = False
         state = self.env.reset()[0]
         while not done:
            action = np.argmax(self.model.predict(state[np.newaxis]))
            logger.info(f"Action: {action}")
            next_state, _, done, _, _ = self.env.step(action)
            self.env.render()
            state = next_state


##########################
###  ARGUMENT PARSING  ###
##########################

class Parsing:
   def __init__(self, args: argparse.Namespace) -> None:
      self.args : dict = args
      self.model : tf.keras.Model = None
      self.data : dict = {}
      for key, value in vars(args).items():
            self.data[key] = value
            
      self.data['x'] = []
      self.data['y'] = []

      if args.listenvs is True:
         envs = [key for key in gym.envs.registration.registry.keys()]
         logger.info(f"List of valid environments ({len(envs)}):")
         for index in range(0, len(envs), 3):
            if len(envs) - index > 2:
               logger.info(f"{str(envs[index])}, {str(envs[index + 1])}, {str(envs[index + 2])}")
            elif len(envs) - index > 1:
               logger.info(f"{str(envs[index])}, {str(envs[index + 1])}")
            else:
               logger.info(f"{str(envs[index])}")
         exit(0)

      if args.uuid is not None:
         if not os.path.exists(f'./models/{self.data["uuid"]}'):
            logger.info(f"The Model at ./models/{self.data['uuid']} does not exist. Exiting...")
            exit(1)

###############
###  MAIN   ###
###############

if __name__ == '__main__':
   logger = Logging().get_logger()
   args = Arguments().parse_args()
   parser = Parsing(args)
   
   if parser.data['play'] is False:
      FileOutputHandler = logging.FileHandler(f"{parser.data['network']}-{parser.data['environment'].translate(str.maketrans('', '', string.punctuation))}.log", "w", encoding="UTF-8")
      logger.addHandler(FileOutputHandler)   
   
   agent = Agent(parser.data)
   agent.run()
   logger.info("Training complete")
   exit(0)
