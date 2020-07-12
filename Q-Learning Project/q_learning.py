#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 22:00:30 2019

@author: davitmelkonyan
"""

import numpy as np

gamma = 0.75
alpha = 0.9

#define the environment
#define the states
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11} #dictionary mapping

actions = [0,1,2,3,4,5,6,7,8,9,10,11]
#build matrix of rewards (2d array)
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
             [1,0,1,0,0,1,0,0,0,0,0,0],
             [0,1,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,0,0,0,0,1,0,0,0],
             [0,1,0,0,0,0,0,0,0,1,0,0],
             [0,0,1,0,0,0,1,1,0,0,0,0],
             [0,0,0,1,0,0,1,0,0,0,0,1],
             [0,0,0,0,1,0,0,0,0,1,0,0],
             [0,0,0,0,0,1,0,0,1,0,1,0],
             [0,0,0,0,0,0,0,0,0,1,0,1],
             [0,0,0,0,0,0,0,1,0,0,1,0]])


   
#mapping from state to location
state_to_loc = {state: loc for loc, state in location_to_state.items()}
#production
def route(start_loc, end_loc):
    R_new = np.copy(R)
    ending_state = location_to_state[end_loc]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12,12]))

    #learning process
    for i in range(0,1000):
        current_state = np.random.randint(0,12)#upper bound is excluded
        playable_actions = []
        for j in range(12): #each of the 12 columns
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
    #learning process end
    route = [start_loc]
    next_loc = start_loc
    while (next_loc != end_loc):
        start_state = location_to_state[start_loc]
        next_state = np.argmax(Q[start_state,])
        next_loc = state_to_loc[next_state]
        route.append(next_loc)
        start_loc = next_loc
    return route

def best_route(start_loc, intermediary_loc, ending_loc):
    return route(start_loc, intermediary_loc) + route(intermediary_loc,ending_loc)[1:]

print('Rout: ')
print (best_route('E','F','G'))






















