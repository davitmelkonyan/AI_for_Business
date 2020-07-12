import numpy as np

class DQN(object):
    def __init__(self, max_memory = 100, discount = 0.9):
       self.memory = list()
       self.max_memory = max_memory
       self.discount = discount 
        

    #Builds the memory in experience replay
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]


    #Builds two batches of 10 inputs and 10 targets by extracting 10 transitions
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1] #transition has at index 0:currState, 1:actionPlayed, 2:reward, 3:nextState
        num_outputs = model.output_shape[-1] #-1 is index of last element
        inputs_batch = np.zeros(min(len_memory, batch_size), num_inputs)#10 rows and 3 cols (num of users, data rate, server temp)
        targets = np.zeros(min(len_memory, batch_size), num_outputs) #10 rows again and now 5 cols/actions
        for i,idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))): #i will take size
            current_state, action, reward, next_state = self.memory[idx][0] # got the transition (not game over)
            game_over = self.memory[idx][1]
            inputs_batch[i] = current_state
            targets[i] = model.predict(current_state)[0] #first index is predicted q value
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs_batch, targets
