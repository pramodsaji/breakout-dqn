# Import necessary libraries
import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import model_from_json
import cv2
from model import create_model
import json

# Class to create a DQN agent
class DQNAgent:
    # Default lives for breakout is 5
    lives = 5

    # Initialise the DQN agent
    # env_name: name of the environment
    def __init__(self, env_name):
        # Set the environment name
        self.env_name = env_name  

        # Create the gym environment with render mode set to human     
        self.env = gym.make(env_name, render_mode = "human")

        # Number of actions from the created environment
        self.action_size = self.env.action_space.n

        # Number of episodes
        self.EPISODES = 5000
        
        # Instantiate memory
        memory_size = 50000
        self.memory = deque(maxlen=memory_size)

        # Discount rate
        self.gamma = 0.99
        
        # Exploration hyperparameters for epsilon strategy
        # Initial exploration probability
        self.epsilon = 1.0

        # Minimum exploration probability
        self.epsilon_min = 0.02

        # Decay rate for exponential decay of exploration probability
        self.epsilon_decay = 0.00002
        
        # Batch size set to 32
        self.batch_size = 32
        
        # Path to save the model
        self.Save_Path = 'Models/ALE'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.Model_name = os.path.join(self.Save_Path, "Breakout-v5_DQN"+".h5")

        # Create list to hold episodes, scores, rewards and average and changing epsilon
        self.episodes, self.scores, self.rewards, self.average, self.epsilons = [], [], [], [], []

        # Set the size of the image frame
        self.ROWS = 80
        self.COLS = 80

        # Set the number of frames to be stacked
        self.REM_STEP = 4
        
        # Set the state size
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)

        # Initialize image memory with zeros
        self.image_memory = np.zeros(self.state_size)
        
        # Create the model with necessary parameters
        self.model = create_model(input_shape=self.state_size, action_space = self.action_size)

    # Function to append experiences to memory
    # state: current state of the environment
    # action: action taken in current state
    # reward: reward received after taking action
    # next_state: next state of the environment after taking action
    # done: whether the episode is complete or not
    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        # Append experience to memory
        self.memory.append((experience))

    # Function to act based on Epsilon-greedy strategy
    # state: current state of the environment
    # decay_step: number of steps taken so far
    def act(self, state, decay_step):
        # Epsilon-greedy strategy
        if self.epsilon > self.epsilon_min:
            # Set epsilon probability value
            self.epsilon *= (1-self.epsilon_decay)
        explore_probability = self.epsilon
    
        # If explore probability is greater than a random number
        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state, verbose = 0)), explore_probability

    # Function to replay experiences and train the agent     
    def replay(self):
        # Start training only if certain number of samples is already saved
        if len(self.memory) > self.batch_size:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, self.batch_size)
        else:
            return
        
        # Initialize the states, actions, rewards, next states and done flags
        state = np.zeros((self.batch_size, *self.state_size), dtype=np.float32)
        action = np.zeros(self.batch_size, dtype=np.int32)
        reward = np.zeros(self.batch_size, dtype=np.float32)
        next_state = np.zeros((self.batch_size, *self.state_size), dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.uint8)

        # Set the values of states, actions, rewards, next states and done flags from the minibatch
        for i in range(len(minibatch)):
            state[i], action[i], reward[i], next_state[i], done[i] = minibatch[i]

        # Predict the target Q-values from the state
        target = self.model.predict(state, verbose = 0)

        # Predict the target Q-values from the next state
        target_next = self.model.predict(next_state, verbose = 0)

        # Loop thru the minibatch and create target Q-values
        for i in range(len(minibatch)):
            # Choose the max Q value among next actions
            target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])
                
        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # Function to update and save the metrics
    # score: score received after taking action
    # reward: reward received after taking action
    # episode: current episode
    def update_save_metrics(self, score, reward, episode):
        self.scores.append(score)
        self.rewards.append(reward)
        self.episodes.append(episode)
        self.epsilons.append(self.epsilon)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        
        # Create a dictionary with the specified keys
        data = {
            "episodes": self.episodes,
            "rewards": self.rewards,
            "epsilons": self.epsilons,
            "scores": self.scores,
            "averages": self.average
        }

        # Open a file in write mode
        with open('metrics.json', 'w') as file:
            json.dump(data, file)

    # Function to preprocess, crop and resize the image
    # frame: current frame of the environment
    def get_image(self, frame):
        # Render the environment
        self.env.render()
        
        # Crop the image to 80x80
        frame_cropped = frame[35:195:2, ::2,:]

        # Resize the image, if not already
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB using NumPy
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # Divide by 255 to transform to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # Push the data by 1 frame, similar to deque
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        # Insert new frame to free space
        self.image_memory[0,:,:] = new_frame

        # Return the image memory
        return np.expand_dims(self.image_memory, axis=0)

    # Function to reset the environment
    def reset(self):
        # Reset the environment and get the initial state
        frame = self.env.reset()
        # Set the image memory with 4 frames of the initial state
        for i in range(self.REM_STEP):
            state = self.get_image(frame[0])
        # Return the initial state
        return state

    # Function to take next step in the environment
    # action: action taken in current state
    def step(self,action):
        # Take the next step in the environment
        next_state, reward, done, _, info = self.env.step(action)
        # Set the reward as score
        score = reward

        # If the agent loses a life, set the reward to -5
        if(info['lives'] != self.lives):
            reward = -5.0
        # Update the lives
        self.lives = info['lives']

        # Preproess the next state
        next_state = self.get_image(next_state)

        # Return the next state, reward, score, done and info
        return next_state, reward, score, done, info
    
    # Function to load the model
    # name: name of the model
    def load(self, name):
        # Load the model architecture from the JSON file
        with open(f'{name}.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)

        # Load the weights into the model
        self.model.load_weights(f'{name}.h5')

    # Function to save the model
    # name: name of the model
    def save(self, name):
        # Serialize model to JSON
        model_json = self.model.to_json()
        # Write to file
        with open(f"{name}.json", "w") as json_file:
            # Write the model to the file
            json_file.write(model_json)
        # Serialize weights to HDF5
        self.model.save_weights(name)

    # Function to run the agent
    def run(self):
        # Initialise the decay step
        decay_step = 0
        # Run for the number of episodes
        for e in range(self.EPISODES):
            print("Episode: ", e)
            # Reset the environment
            state = self.reset()
            # Set done to False
            done = False
            # Set the overall score and overall reward to 0
            o_score = 0
            o_reward = 0

            # Run till the episode is complete
            while not done:
                # Increment the decay step
                decay_step += 1
                # Get the action and explore probability
                action, explore_probability = self.act(state, decay_step)
                # Take the next step in the environment
                next_state, reward, score, done, _ = self.step(action)
                # Add the experience to memory
                self.remember(state, action, reward, next_state, done)
                # Set the state to next state
                state = next_state

                # Update the overall score and reward
                o_score += score
                o_reward += reward

                # If the episode is complete
                if done:
                    # Set the lives to 5
                    self.lives = 5.0
                    
                    # Save the model
                    self.save(self.Model_name)

                    # Update and save the metrics
                    self.update_save_metrics(o_score, o_reward, e)

                    print("Episode: {}/{}, Score: {}, Reward: {}, e: {:.2f}, Average: {:.2f}".format(e, self.EPISODES, o_score, o_reward, explore_probability, self.average[-1]))

                # Train the agent with the experience
                self.replay()

        # Close the environment after running the agent
        self.env.close()

    # Function to test the agent
    # model_name: name of the model
    def test(self, model_name):
        # Load the model
        self.load(model_name)
        # Run for the number of episodes
        for e in range(self.EPISODES):
            # Reset the environment
            state = self.reset()
            # Set done flag to False
            done = False
            # Set the overall score to 0
            o_score = 0
            # Run till the episode is complete
            while not done:
                # Render the environment
                self.env.render()
                # Get the action
                action = np.argmax(self.model.predict(state, verbose = 0))
                # Take the next step in the environment
                state, reward, score, done, _ = self.step(action)
                # Update the overall score
                o_score += score
                # If the episode is complete
                if done:
                    print("Episode: {}/{}, Score: {}".format(e, self.EPISODES, o_score))
                    break
        # Close the environment after testing the agent
        self.env.close()