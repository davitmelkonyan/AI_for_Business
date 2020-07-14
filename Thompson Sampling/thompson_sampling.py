import numpy as np
import matplotlib.pyplot as plt
import random

#params
N = 10000 #rounds
d = 9 #strategies to choose from

#conversion_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] - using these rates, more ones will be to the right and more 0s to the left
conversion_rates = [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]
X = np.array(np.zeros([N,d]))
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i,j] = 1

#Thompson Sampling

#beta distribution for each 9 strategies
strategies_selected_rs = [] #rs = random strategy
strategies_selected_ts = [] #ts = thomspon sampling
total_reward_rs = 0
total_reward_ts = 0
numbers_of_rewards_1 = [0] * d #list of 9 0s -> num of times strategy i recieved reward 1 up to round n
numbers_of_rewards_0 = [0] * d #list of 9 0s -> num of times strategy i recieved reward 0 up to round n

for n in range(0, N):
    # Random Strategy
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs = total_reward_rs + reward_rs

    #Thompson Sampling
    #---------STEP1------------ for each startegy i, take random draw from beta distribution
    strategy_ts = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,numbers_of_rewards_0[i] + 1)
        #---------STEP2------------ select strategy s(n) that has highest random draw
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i
    #---------STEP3---------------- update num of times strategies have recieved 1 or 0 reward up to round N
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts = total_reward_ts + reward_ts

#Computing the Absolute and Relative (the return an asset achieves over period of time compared to a benchmark, for us random strategy->i.e assets return - benchmarks return) return
absolute_return = (total_reward_ts - total_reward_rs)*100
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Absolute Return: {:.0f} $".format(absolute_return))
print("Relative Return: {:.0f} %".format(relative_return))

#Plot the histogram of seleection
plt.hist(strategies_selected_ts)
plt.title("Histogram of Selections")
plt.xlabel("Strategy")
plt.ylabel("Number of times the strategy was selected")
plt.show()