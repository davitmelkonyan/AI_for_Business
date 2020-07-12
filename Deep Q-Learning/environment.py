import numpy as np

class Environment(object):
    #initialize all params and variables
    def __init__(self, optimal_temp = (18.0,24.0), 
                 initial_month = 0, 
                 initial_number_users = 10,
                 initial_rate_data = 60):
        self.monthly_atmospheric_temp = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temp = self.monthly_atmospheric_temp[initial_month]
        self.optimal_temp = optimal_temp
        self.min_temp = -20
        self.max_temp = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temp
        self.temperature_no_ai = (self.optimal_temp[0]+self.optimal_temp[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_no_ai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    #update the environment
    def update_env(self, direction, energy_ai, month):
        #GET THE REWARD
        #compute energy spent with no AI
        energy_no_ai = 0
        if (self.temperature_no_ai < self.optimal_temp[0]):
            energy_no_ai = self.optimal_temp[0] - self.temperature_no_ai
            self.temperature_no_ai = self.optimal_temp[0]
        elif (self.temperature_no_ai > self.optimal_temp[1]):
            energy_no_ai = self.temperature_no_ai - self.optimal_temp[1]
            self.temperature_no_ai = self.optimal_temp[1]
        #compute the reward
        self.reward = energy_no_ai - energy_ai
        #scale the reward
        self.reward = 1e-3 * self.reward

        #-------GET THE NEXT STATE------

        #update atmosph temp.
        self.atmospheric_temp = self.monthly_atmospheric_temp[month]
        #update number of users
        self.current_number_users += np.random.randint(-self.max_update_users,self.max_update_users)
        if(self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users  < self.min_number_users):
            self.current_number_users = self.min_number_users
        self.current_rate_data += np.random.randint(-self.max_update_data,self.max_update_data)
        if(self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data  < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        #compute the delta
        past_intrinsic_temp = self.intrinsic_temp
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temp = self.intrinsic_temp - past_intrinsic_temp
        #delta of temp caused by AI
        if(direction == -1):#cools down
            delta_temp_ai = -energy_ai
        elif (direction == 1):
            delta_temp_ai = energy_ai
        self.temperature_ai += delta_intrinsic_temp  + delta_temp_ai
        #no AI
        self.temperature_ai += delta_intrinsic_temp

        #-------GET the GAME OVER------
        if(self.temperature_ai < self.min_temp):
            if(self.train == 1):#in training mode
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temp[0]
                self.total_energy_ai += self.optimal_temp[0] - self.temperature_ai
        elif (self.temperature_ai > self.max_temp):
            if(self.train == 1):#in training mode
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temp[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temp[1]
        
        #--------UPDATE THE SCORES-----------
        #update totak energy spent by ai
        self.total_energy_ai += energy_ai
        #same for no_ai
        self.total_energy_no_ai +=energy_no_ai

        #---------SCALE THE NEXT STATE-------
        scaled_temp_ai = (self.temperature_ai - self.min_temp)/(self.max_temp - self.min_temp)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temp_ai,scaled_number_users,scaled_rate_data])#the vector
        
        #----------RETURN-------
        return next_state, self.reward, self.game_over
        
    #Resets the environment
    def reset(self, new_month ):# reset the variables
        self.atmospheric_temp = self.monthly_atmospheric_temp[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temp
        self.temperature_no_ai = (self.optimal_temp[0]+self.optimal_temp[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_no_ai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    #gives current state, the last reward
    def observe(self):
        scaled_temp_ai = (self.temperature_ai - self.min_temp)/(self.max_temp - self.min_temp)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temp_ai,scaled_number_users,scaled_rate_data])#the vector
        return current_state, self.reward, self.game_over
 
    