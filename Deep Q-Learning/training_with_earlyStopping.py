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
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

#build environment object
env = environment.Environment(optimal_temp = (18.0,24.0), 
                              initial_month = 0, 
                              initial_number_users = 20,
                              initial_rate_data = 30)
#build brain obj
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)
#build dqn obj
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

#choose the mode
train = True

#--------START TRAINING----------
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if(env.train):
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0,12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _,_ = env.observe() #only get the first thing observe method returns from 3
        timestep = 0
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):#exploration vs explortations
            #next action by exploation (30% of time)
            if np.random.rand() <= epsilon: #exploration
                action = np.random.randint(0,number_actions)
                if (action - direction_boundary < 0):#dir of temp change
                    direction = -1 #cooling down
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step #check formula for all cases - done
            else: #exploitation - next action by inference (70% of time)
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])#action coresponding to highest q values
                if (action - direction_boundary < 0):#dir of temp change
                    direction = -1 #cooling down
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            #update the environment and reach next step
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            #Store this new transition into memory
            dqn.remember([current_state,action, reward, next_state],game_over)
            #two separate batches
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            #compute the loss over the two batches with MINIBATCH gradient descent (combination of stochastoc gradiet descent and batch learning)
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        #print training results
        print ("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_no_ai))

        #--------EARLY STOPPING---------
        if(early_stopping):
            if (total_reward <= best_total_reward):
                patience_count +=1
            elif(total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count >= patience):
                print("Early Stopping")
                break
        #--------SAVE THE MODEL---------
        model.save("model.h5")