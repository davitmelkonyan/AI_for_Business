import os #setting seeds for reproducibility i.e. to get same ans as in the course
import numpy as np
import random as rn
#my libs
import environment
import brain
import dqn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#setting parameters
epsilon = 0.3 #exploration parameter (do random actions 30% of time)
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 1000
max_memory = 3000
batch_size = 512
temperature_step = 1.5

#build environment object
env = environment.Environment(optimal_temp = (18.0,24.0), 
                              initial_month = 0, 
                              initial_number_users = 20,
                              initial_rate_data = 30)
#build brain obj
brain = brain.Brain(learning_rate = 0.0001, number_actions = number_actions)
#build dqn obj
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

#choose the mode
train = True

#--------START TRAINING----------
env.train = train
model = brain.model
if(env.train):
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0,12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _,_ = env.observe #only get the first thing observe method returns from 3
        

