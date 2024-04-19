import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Data Generation
np.random.seed(42)  # Ensure reproducibility
balances_data = {
    'Client 1': np.append(np.random.lognormal(mean=5, sigma=0.8, size=20), [800, 1500]),
    'Client 2': np.append(np.random.lognormal(mean=6, sigma=0.9, size=20), [1200, 2000]),
    'Client 3': np.append(np.random.lognormal(mean=5.5, sigma=0.7, size=20), [700, 1000]),
}
# Iteration 1: Basic components scaled between 0 and 1
def loss_function_v1(level, balances):
    total_difference = np.sum(np.abs(balances - level)) / np.sum(np.abs(balances - np.mean(balances)))
    variance = np.var(balances - level) / np.var(balances)
    below_optimal_penalty = np.sum(balances < level) / len(balances)
    changes_across_penalty = np.sum(np.abs(np.diff(balances - level > 0))) / (len(balances) - 1)
    return total_difference + variance + below_optimal_penalty + changes_across_penalty

# Iteration 2: Increase penalty for below optimal
def loss_function_v2(level, balances):
    total_difference = np.sum(np.abs(balances - level)) / np.sum(np.abs(balances - np.mean(balances)))
    variance = np.var(balances - level) / np.var(balances)
    below_optimal_penalty = (np.sum(balances < level) / len(balances)) ** 2  # Squared to increase penalty
    changes_across_penalty = np.sum(np.abs(np.diff(balances - level > 0))) / (len(balances) - 1)
    return total_difference + variance + below_optimal_penalty + changes_across_penalty

# Iteration 3: Adjusted penalty for changes across optimal level
def loss_function_v3(level, balances):
    total_difference = np.sum(np.abs(balances - level)) / np.sum(np.abs(balances - np.mean(balances)))
    variance = np.var(balances - level) / np.var(balances)
    below_optimal_penalty = (np.sum(balances < level) / len(balances)) ** 2
    changes_across_penalty = (np.sum(np.abs(np.diff(balances - level > 0))) / (len(balances) - 1)) ** 2  # Squared to increase penalty
    return total_difference + variance + below_optimal_penalty + changes_across_penalty

# Execution and comparison framework for iterating over loss function versions
def execute_loss_function_iterations(balances_data):
    results = {}
    functions = [loss_function_v1, loss_function_v2, loss_function_v3]
    iteration_labels = ["Iteration 1", "Iteration 2", "Iteration 3"]
    
    for func, label in zip(functions, iteration_labels):
        optimal_levels = {}
        for client, balances in balances_data.items():
            result = minimize(func, x0=np.median(balances), args=(balances,),
                              method='SLSQP', bounds=[(np.min(balances), np.max(balances))])
            optimal_levels[client] = result.x[0]
        results[label] = optimal_levels

    return results

# Visualization of results for each iteration
def visualize_optimization_results(balances_data, results):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (client, balances) in enumerate(balances_data.items()):
        axs[i].scatter(range(len(balances)), balances, label='Balances', alpha=0.6)
        axs[i].axhline(y=np.mean(balances), color='red', label='Mean', linestyle='--')
        axs[i].axhline(y=np.median(balances), color='orange', label='Median', linestyle='-.')
        for label, optimal_levels in results.items():
            axs[i].axhline(y=optimal_levels[client], linestyle='--', label=f'{label}: {optimal_levels[client]:.2f}')
        axs[i].set_title(client)
        axs[i].set_xlabel('Day')
        axs[i].legend()
        axs[i].set_ylabel('Balance' if i == 0 else '')
    plt.suptitle('Optimal Levels Across Iterations')
    plt.tight_layout()
    plt.show()

# Perform the execution of iterations and visualize the results
results = execute_loss_function_iterations(balances_data)
visualize_optimization_results(balances_data, results)

# Analysis and selection of the best loss function version will be based on the visualized results and the objectives.






# # Step 2: Optimization Process
# def objective_function(balance, balances):
#     # Custom objective function to minimize
#     return np.sum(np.abs(balances - balance)) + np.var(balances - balance) * 100
# def corrected_objective_function(level, balances):
#     # Calculate total absolute difference
#     total_difference = np.sum(np.abs(balances - level))
#     print('total_difference',total_difference)
#     # Calculate variance
#     variance = np.var(balances - level)
#     print('variance',variance)
#     # Penalty for balances below the optimal level
#     below_optimal_penalty = np.sum(balances < level) * 100  # Weight can be adjusted
#     print('below_optimal_penalty',below_optimal_penalty/100)
#     # Penalty for changes across the optimal level
#     changes_across_penalty = np.sum(np.diff(balances - level > 0)) * 100  # Weight can be adjusted
#     print('changes_across_penalty',changes_across_penalty/100)
    
#     # Combine all components into the total loss
#     total_loss = total_difference + variance * 100 + below_optimal_penalty + changes_across_penalty
#     return total_loss
# optimal_levels = {}
# optimal_levels_new = {}


# for client, balances in clients_data.items():
#     initial_guess = np.median(balances)
#     bounds = [(np.min(balances), np.max(balances))]
#     result = minimize(objective_function, x0=initial_guess, args=(balances,), method='SLSQP', bounds=bounds)
#     result1 = minimize(corrected_objective_function, x0=initial_guess, args=(balances,), method='SLSQP', bounds=bounds)

#     optimal_levels[client] = result.x[0]
#     optimal_levels_new[client] = result1.x[0]

# # Step 3: Visualization
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# for i, (client, balances) in enumerate(clients_data.items()):
#     axs[i].scatter(range(22), balances, color='blue', label='Balances')
#     axs[i].axhline(y=np.mean(balances), color='red', label='Mean', linestyle='--')
#     axs[i].axhline(y=np.median(balances), color='orange', label='Median', linestyle='-.')
#     axs[i].axhline(y=optimal_levels[client], color='green', label='Optimized', linestyle='-')
#     axs[i].axhline(y=optimal_levels_new[client], color='black', label='Optimized_new', linestyle='-')
#     axs[i].set_title(client)
#     axs[i].legend()

# plt.tight_layout()
# plt.show()

# # Step 4: Analysis and Output
# print("Optimized Balance Levels:")
# for item, items1 in zip(optimal_levels.items(),optimal_levels_new.items()):
#     print(f"{item}: {items1} ")
# # for client, level,client1, level1 in zip(optimal_levels.items(),optimal_levels_new.items()):
#     # print(f"{client}: {level:.2f}, {level1:.2f} : {client1} ")
