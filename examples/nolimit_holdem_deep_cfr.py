''' An example of solve Leduc Hold'em with CFR
'''
import numpy as np

import rlcard
import tensorflow as tf
from rlcard import models
from rlcard.agents.deep_cfr_agent import DeepCFR
from rlcard.utils.utils import set_global_seed, tournament
from rlcard.utils.logger import Logger
from rlcard.agents.random_agent import RandomAgent

# Make environment and enable human mode
env = rlcard.make('no-limit-holdem', config={'allow_step_back':True})
eval_env = rlcard.make('no-limit-holdem')

# Set the iterations numbers and how frequently we evaluate the performance and save model
evaluate_every = 10
save_plot_every = 10
evaluate_num = 10
episode_num = 50

# The paths for saving the logs and learning curves
log_dir = './experiments/no-limit-holdem_deep_cfr_result/'

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:
    # Initilize CFR Agent
    agent = DeepCFR(session=sess, env=env)
    # agent.load()  # If we have saved model, we first load the model
    # agent = models.load('nolimit_holdem_dqn').agents[0]
    # agent_random = RandomAgent(action_num=env.action_num)
    agent2 = models.load('nolimit_holdem_dqn').agents[0]
    eval_env.set_agents([agent, agent2])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with NFSP agents.
        if episode % evaluate_every == 0:
            # agent.save() # Save model
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('DeepCFR')
