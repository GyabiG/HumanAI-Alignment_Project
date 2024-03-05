import numpy as np
import gym
import pygame
from teleop import collect_demos
from scipy.optimize import minimize
import maxent_IRL as me



#collection of demos
def collect_human_demos(env, num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

# Collect expert demonstrations using a random policy
def generate_expert_trajectories(env, num_trajectories=10, max_steps=200):
    expert_trajectories = []

    for _ in range(num_trajectories):
        demos = collect_human_demos(env, 1)

        # Convert demos to trajectories
        trajectory = []
        for act in demos:
            state = selectState((act[0][0], act[0][1])) # Assuming state is a tuple of (position, velocity)
            action = act[1]
            trajectory.append((state, action))

        expert_trajectories.append(trajectory)

    return expert_trajectories

def selectState(obs):
    state = 0
    if obs[0] < -0.3:
        if obs[1] < 0:
            state = 1
        if obs[1] >= 0:
            state = 2
    else:
        if obs[1] < 0:
            state = 3
        if obs[1] >= 0:
            state = 4
    #print('obs', obs, 'state:', state)
    return state
#evaluate learned policyS
def evaluate_policy(policy, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0", render_mode='human')
    else:
        env = gym.make("MountainCar-v0")

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        #print("obs: ", obs)
        while not done:
            # Take action based on the probabilities in the policy array
            action = np.array(policy[selectState(obs)]).argmax()
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))



def main():
    env = gym.make("MountainCar-v0",render_mode='rgb_array') 

    expert_trajectories = generate_expert_trajectories(env, 2)


    #print(expert_trajectories)
    # Set up MaxEnt IRL model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    feature_dim =  2 # Example feature dimension for Mountain Car
    irl_model = me.MaxEntIRL(env, feature_dim, 3, expert_trajectories)
    irl_model.train()

    # Get the learned policy
    learned_policy = irl_model.get_learned_policy()
    print("Learned Policy:")
    print(learned_policy)

    evaluate_policy(learned_policy, 5)

if __name__ == "__main__":
    main()
