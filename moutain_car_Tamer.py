import numpy as np
import gym
import keyboard
import pygame
from teleop import collect_demos
from scipy.optimize import minimize
import maxent_IRL as me
from moutain_car_IRL import evaluate_policy, selectState



def tamer_algorithm(policy, num_evals):
    env = gym.make("MountainCar-v0", render_mode='human')


    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()

        while not done:
            # Take action based on the probabilities in the policy array
            action = np.array(policy[selectState(obs)]).argmax()
            obs, rew, done, info = env.step(action)
            total_reward += rew

            # Check for space bar press to update policy
            if keyboard.is_pressed('Space'):
                # Tamer algorithm: Update policy based on the observed state
                #print(policy[selectState(obs)])
                if policy != 0:
                    currPolicy = policy[selectState(obs)]
                    policy[selectState(obs)] = updatePolicy(currPolicy[0], action) 


        print("Reward for Tamer", i, total_reward)
        policy_returns.append(total_reward)

    env.close()
    return policy

def updatePolicy(currPolicy, action):
    newPolicy = currPolicy
    i = 0.9
    d = 0
    #print('curr', currPolicy)
    #print('action', action)
    if action == 0:
        newPolicy[0] = currPolicy[0] * i
    if action == 1:
        newPolicy[1] = currPolicy[1] * i
    else:
        newPolicy[2] = currPolicy[2] * i

    nf = 1/np.sum(np.array(newPolicy))
    print(np.sum(newPolicy))
    newPolicy[0] = newPolicy[0]*nf
    newPolicy[1] = newPolicy[1]*nf
    newPolicy[2] = newPolicy[2]*nf
    return [newPolicy]

def main():
    env = gym.make("MountainCar-v0", render_mode='rgb_array')
    #Sample policies have been previously generated from moutain_car_IRL
    sample_policy1 = [0, ([[0.44060333, 0.13706362, 0.42233305]]), ([[0.39434542, 0.09266859, 0.51298599]]), ([[0.33979577, 0.06031913, 0.5998851 ]]), ([[0.28328513, 0.03798763, 0.67872724]])]
    sample_policy2 = [0, ([[0.25231694, 0.44672479, 0.30095827]]), ([[0.15815781, 0.4197246 , 0.42211759]]), ([[0.09132434, 0.36327918, 0.54539647]]), ([[0.04919874, 0.29335134, 0.65744992]])]
    learned_policy = tamer_algorithm(sample_policy1, 5)
    print('learned policy:', learned_policy)
    evaluate_policy(learned_policy, 5, env)

if __name__ == "__main__":
    main()
