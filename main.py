# Importing necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from DQNAgent import *

if __name__ == "__main__":
    # Set the env name and create the env    
    env_name = 'ALE/Breakout-v5'
    agent = DQNAgent(env_name)
    # Train the agent
    agent.run()

    # Load saved model and weights
    # model_to_load = 'Models/ALE/Breakout-v5_DQN.h5'
    # agent.test(model_to_load)


