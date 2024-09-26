# import numpy as np

# # Observed correlation coefficient
# observed_corr = 0.3

# # Number of Monte Carlo iterations
# num_iterations = 10000

# # Placeholder for storing correlation coefficients
# simulated_corrs = []

# # Simulate data assuming independence
# for _ in range(num_iterations):
#     # Generate random data for A and S (assuming independence)
#     random_A = np.random.rand(10)  # Generating 10 random values for A
#     random_S = np.random.rand(10)  # Generating 10 random values for S
    
#     # Calculate correlation coefficient between random A and S
#     corr = np.corrcoef(random_A, random_S)[0, 1]
#     simulated_corrs.append(corr)

# # Count occurrences where simulated correlation >= observed correlation
# count = sum(c >= observed_corr for c in simulated_corrs)

# # Compute simulated p-value
# simulated_p_value = count / num_iterations

# print(f"Simulated p-value: {simulated_p_value}")

# from scipy.stats import norm

# # Given parameters
# mu1, sigma1 = 0, 2
# mu2, sigma2 = 2, 3
# sample_size = 1000

# # Sample from f and compute the KL value at each sample point
# X = mu1 + sigma1 * np.random.randn(sample_size)

# pdf_f = norm.pdf(X, mu1, sigma1)
# pdf_g = norm.pdf(X, mu2, sigma2)
# KL = np.log(pdf_f / pdf_g)

# KL_div = np.mean(KL)
# KL_div_std = np.std(KL) / np.sqrt(sample_size)

# # KL divergence based on theoretical expression
# KL_div_theory = np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

# print("KL Divergence (Estimated):", KL_div)
# print("Standard deviation of KL:", KL_div_std)
# print("KL Divergence (Theoretical):", KL_div_theory)

# import numpy as np
# import matplotlib.pyplot as plt

# # Function to simulate the bandit problem
# def bandit(k, epsilon, T, c):
#     q_star = np.random.normal(size=k)  # True means of each arm
#     Q = np.zeros(k)  # Estimated values of each action
#     N = np.zeros(k)  # Number of times each action was chosen
#     total_reward = 0
#     total_regret = []
#     correct_decisions = np.zeros(T)  # Track correct decisions

#     for t in range(1, T + 1):
#         if np.random.rand() < epsilon:
#             action = np.random.randint(k)  # Exploration: choose random action
#         else:
#             action = np.argmax(Q + c * np.sqrt(np.log(t) / (N + 1e-5)))  # UCB action selection

#         reward = np.random.normal(q_star[action])  # Get reward for chosen action
#         total_reward += reward
#         N[action] += 1
#         Q[action] += (reward - Q[action]) / N[action]  # Update action value estimate

#         # Check if the correct arm was chosen
#         if action == np.argmax(q_star):
#             correct_decisions[t - 1] = 1  # Record a correct decision

#         # Calculate regret at each time step
#         regret = np.max(q_star) - q_star[action]
#         total_regret.append(regret)

#     return total_regret, total_reward, correct_decisions

# # Parameters
# k = 5  # Number of arms
# T = 1000  # Total actions
# epsilons = [0.1, 0.2, 0.5]  # List of epsilon values to test
# ucb_parameters = [1, 2, 5]  # List of UCB hyper-parameter values to test
# num_experiments = 1000  # Number of experiments for each strategy/hyper-parameter

# # Calculate percentage optimal choices for epsilon-greedy
# epsilon_optimal_choices = []
# for epsilon in epsilons:
#     avg_optimal_choices = np.zeros(T)
#     for _ in range(num_experiments):
#         _, _, correct_decisions = bandit(k, epsilon, T)
#         avg_optimal_choices += correct_decisions
#     avg_optimal_choices /= num_experiments
#     epsilon_optimal_choices.append(avg_optimal_choices * 100)
#     print(f"Epsilon {epsilon} - Percentage of correct decisions: {np.mean(avg_optimal_choices) * 100:.2f}%")

# # Calculate percentage optimal choices for UCB with different hyper-parameters
# ucb_optimal_choices = []
# for c in ucb_parameters:
#     avg_optimal_choices = np.zeros(T)
#     for _ in range(num_experiments):
#         _, _, correct_decisions = bandit(k, epsilons[0], T, c=c)  # Use the first epsilon for UCB
#         avg_optimal_choices += correct_decisions
#     avg_optimal_choices /= num_experiments
#     ucb_optimal_choices.append(avg_optimal_choices * 100)
#     print(f"UCB with c={c} - Percentage of correct decisions: {np.mean(avg_optimal_choices) * 100:.2f}%")

# # Plotting
# plt.figure(figsize=(10, 6))

# for i, epsilon in enumerate(epsilons):
#     plt.plot(np.arange(1, T + 1), epsilon_optimal_choices[i], label=f'Epsilon={epsilon}, Greedy', linestyle='-', alpha=0.7)

# for i, c in enumerate(ucb_parameters):
#     plt.plot(np.arange(1, T + 1), ucb_optimal_choices[i], label=f'UCB, c={c}', linestyle='--', alpha=0.7)

# plt.xlabel('Number of Actions')
# plt.ylabel('Percentage of Optimal Choices')
# plt.title('Comparison of Strategies: Epsilon-Greedy vs UCB')
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np

# # Payoff matrix
# payoff_matrix = np.array([
#     [[1, 5], [2, 2], [3, 4], [3, 1]],
#     [[3, 0], [4, 1], [2, 5], [4, 2]],
#     [[1, 3], [2, 6], [5, 2], [2, 3]]
# ])

# # Initial strategies (can start with equal probabilities)
# player1_strategy = np.array([1/3, 1/3, 1/3])
# player2_strategy = np.array([1/4, 1/4, 1/4, 1/4])

# # Number of iterations
# iterations = 1000

# for _ in range(iterations):
#     # Player 1's turn
#     expected_values_p1 = np.dot(player2_strategy, payoff_matrix[:, :, 0].T)
#     best_response_p1 = np.argmax(expected_values_p1)
#     player1_strategy = np.zeros_like(player1_strategy)
#     player1_strategy[best_response_p1] = 1

#     # Player 2's turn
#     expected_values_p2 = np.dot(player1_strategy, payoff_matrix[:, :, 1])
#     best_response_p2 = np.argmax(expected_values_p2)
#     player2_strategy = np.zeros_like(player2_strategy)
#     player2_strategy[best_response_p2] = 1

# # Normalize strategies to get probabilities
# player1_strategy /= np.sum(player1_strategy)
# player2_strategy /= np.sum(player2_strategy)

# print("Player 1's mixed strategy:", player1_strategy)
# print("Player 2's mixed strategy:", player2_strategy)



import numpy as np

# Payoff matrix
payoffs = np.array([
    [[1, 5], [2, 2], [3, 4], [3, 1]],
    [[3, 0], [4, 1], [2, 5], [4, 2]],
    [[1, 3], [2, 6], [5, 2], [2, 3]]
])

# Number of iterations
num_iterations = 1000

# Initialize strategy profile estimates for players
player1_strategy = np.zeros(payoffs.shape[0])
player2_strategy = np.zeros(payoffs.shape[1])

# Fictitious Play algorithm
for i in range(num_iterations):
    # Estimate opponent's strategy based on past actions
    player2_empirical = np.sum(player1_strategy) / (i + 1)
    player1_empirical = np.sum(player2_strategy) / (i + 1)

    # Calculate best response for each player
    player1_best_response = np.argmax(np.dot(payoffs[:, :, 0], player2_strategy))
    player2_best_response = np.argmax(np.dot(payoffs[:, :, 1].T, player1_strategy))

    # Update strategy profiles based on best responses
    player1_strategy[player1_best_response] += 1
    player2_strategy[player2_best_response] += 1

# Normalize strategies to get mixed strategies
player1_mixed_strategy = player1_strategy / np.sum(player1_strategy)
player2_mixed_strategy = player2_strategy / np.sum(player2_strategy)

print("Player 1's mixed strategy:", player1_mixed_strategy)
print("Player 2's mixed strategy:", player2_mixed_strategy)
