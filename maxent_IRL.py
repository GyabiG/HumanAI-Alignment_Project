import numpy as np
from scipy.optimize import minimize

class MaxEntIRL:
    def __init__(self, env, feature_dim, action_dim, expert_trajectories, learning_rate=0.01, max_iterations=500):
        self.env = env
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.expert_trajectories = expert_trajectories

        # Initialize reward function parameters randomly
        self.weights = np.random.rand(feature_dim, action_dim)

    def feature_expectations(self, trajectories):
        # Calculate the feature expectations from the expert trajectories
        total_features = np.zeros((self.feature_dim, self.action_dim))
        for trajectory in trajectories:
            for state, action in trajectory:
                features = self.calculate_features(state)
                total_features[:, action] += features
        return total_features / len(trajectories)

    def calculate_features(self, state):
        # Feature function for Mountain Car (position and velocity)
        return np.array([state, 1])

    def entropy(self, policy):
        # Calculate the entropy of the policy
        return -np.sum(policy * np.log(policy + 1e-8))

    def compute_policy(self, logits):
        # Reshape the logits array to ensure it has two dimensions
        logits = logits.reshape(-1, self.action_dim)

        # Compute the softmax policy based on the logits for each action
        exponentiated = np.exp(logits)
        denominator = np.sum(exponentiated, axis=1, keepdims=True)
        policy = exponentiated / (denominator + 1e-8)

        # Return only the first 5 elements of each policy
        return policy[:, :5]

    def objective_function(self, weights):
        # Objective function to be minimized
        feature_expert = self.feature_expectations(self.expert_trajectories)
        all_state_features = self.calculate_all_state_features()

        weights = weights.reshape(self.feature_dim, self.action_dim)
        
        logits = np.dot(all_state_features, weights)
        policy = self.compute_policy(logits)
        feature_learned = np.dot(all_state_features.T, policy)

        # Maximize entropy subject to feature expectations
        return self.entropy(policy) - np.sum(weights * (feature_learned - feature_expert))

    def calculate_all_state_features(self):
        # Calculate features for all states in all trajectories
        all_features = []
        for trajectory in self.expert_trajectories:
            for state, _ in trajectory:
                features = self.calculate_features(state)
                all_features.append(features)
        return np.array(all_features)

    def train(self):
        # Use optimization to find the optimal reward function parameters
        result = minimize(self.objective_function, self.weights.flatten(), method='L-BFGS-B',
                          options={'maxiter': self.max_iterations})
        self.weights = result.x.reshape((self.feature_dim, self.action_dim))

    def get_learned_policy(self):
        # Get the learned policy based on the trained reward function
        all_state_features = self.calculate_all_state_features()
        logits = np.dot(all_state_features, self.weights)
        # Return a list of tuples where each tuple contains a state and its corresponding policy
        policy = [0,0,0,0,0]
        long_policy = [(state, self.compute_policy(logit)) for state, logit in zip(all_state_features, logits)]
        for p in long_policy:
            policy[p[0][0]] = p[1]
        return policy
