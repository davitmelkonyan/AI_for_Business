import os #setting seeds for reproducibility i.e. to get same ans as in the course
import numpy as np
import random as rn
from keras.models import load_model
#my libs
import environment

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#setting parameters
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5 

#build environment object
env = environment.Environment(optimal_temp = (18.0,24.0), 
                              initial_month = 0, 
                              initial_number_users = 20,
                              initial_rate_data = 30)
#Load pre-trained brain
model = load_model("model.h5")#thanks keras

#choose the mode
train = False

#--------RUNNING 1 YEAR SIMULATION IN INFERENCE MODE----------
env.train = train
current_state, _,_ = env.observe()
for timestep in range(0, 12* 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])#action coresponding to highest q values
    if (action - direction_boundary < 0):#dir of temp change
        direction = -1 #cooling down
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state

#print training results
print ("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_no_ai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_no_ai - env.total_energy_ai)/env.total_energy_no_ai*100))