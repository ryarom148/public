# Since the execution state was reset, we'll need to redefine the necessary imports, synthetic data generation, and the provided loss function before applying differential evolution optimization.

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Re-generate synthetic daily balances for three clients with non-normal distributions and outliers
np.random.seed(42)  # Ensure reproducibility
client_data = {
    'Client 1': np.append(np.random.lognormal(mean=5.5, sigma=0.5, size=20), [1000, 1500]),
    'Client 2': np.append(np.random.lognormal(mean=6, sigma=0.6, size=20), [2000, 2500]),
    'Client 3': np.append(np.random.lognormal(mean=5, sigma=0.4, size=20), [800, 1200]),
}
# Example usage
#client_data = {'Client 1': np.array([100, 120, 95, 110, 80, 135, 105, 90, 140, 115, 75, 125, 85, 100, 98, 150, 120, 88, 112, 108, 92, 130])}
# Define the loss function as provided
def calculate_profit_or_loss(optimal_level, balances, monthly_rate=0.06, borrowing_rate=0.05, lending_rate=0.03):
    """
    Calculates the profit or loss based on optimal level, actual balances, and interest rates.

    Args:
        optimal_level (float): The optimal balance level determined through optimization.
        balances (np.ndarray): Array of actual daily balances.
        monthly_rate (float): Monthly interest rate for investing the optimal level.
        borrowing_rate (float): Daily borrowing rate for when actual balance is below optimal.
        lending_rate (float): Daily lending rate for when actual balance is above optimal.

    Returns:
        float: Total profit or loss after 22 days.
    """

    # Convert monthly rate to daily for consistency
    daily_investment_rate = (monthly_rate /12)/30
    daily_borrowing_rate = (borrowing_rate/12)/30
    daily_lending_rate = (lending_rate/12)/30
    # Initial investment profit from investing the optimal level for 22 days
    investment_profit = optimal_level * daily_investment_rate * 22

    daily_profit_or_loss = 0
    for balance in balances:
        if balance < optimal_level:
            # Borrow the difference at borrowing_rate
            daily_profit_or_loss += (balance-optimal_level) * daily_borrowing_rate
        elif balance > optimal_level:
            # Invest the difference at lending_rate
            daily_profit_or_loss += (balance - optimal_level) * daily_lending_rate

    total_profit_or_loss = investment_profit + daily_profit_or_loss
    return total_profit_or_loss

# Sample rates for execution
monthly_rate = 0.05  # 1% annualised
borrowing_rate = 0.04  # annualised borrowing rate
lending_rate = 0.03  # annualised lending rate


def combined_loss(level, balances):
    n = len(balances)
    max_diff = np.max(np.abs(balances - np.mean(balances)))  # Max difference for scaling

    total_difference = np.sum(np.abs(balances - level)) / (n * max_diff)
    variance_ratio = np.var(balances - level) / max_diff**2
    below_optimal = np.sum(balances < level) / n
    changes_across = np.sum(np.abs(np.diff(balances - level > 0))) / (n - 1)

    loss = total_difference + variance_ratio + below_optimal + changes_across
    return loss
def combined_loss1(level, balances):
    n = len(balances)
    max_diff = np.max(np.abs(balances - np.mean(balances)))  # Max difference for scaling

    #total_difference = np.sum(np.abs(balances - level)) / (n * max_diff)
    total_difference = np.sum(np.abs(balances - level)) / np.sum(np.abs(balances - np.mean(balances)))
    variance = np.var(balances - level) / np.var(balances)
    variance_ratio = np.var(balances - level) /  np.var(balances)
    below_optimal = np.sum(balances < level) / n
    changes_across = np.sum(np.abs(np.diff(balances - level > 0))) / (n - 1)

    loss = total_difference + variance_ratio + below_optimal + changes_across
    return loss
# Apply differential evoluti
def quantile_loss(level, balances, quantile=0.1):
  """
  Loss function based on quantile and shortfall below the level.

  Args:
      level (float): The level to evaluate for liquidity risk.
      balances (np.ndarray): A 1D array of daily balances.
      quantile (float, optional): The quantile threshold (defaults to 0.1).

  Returns:
      float: The combined loss value.
  """

  shortfall = np.maximum(0, level - np.quantile(balances, quantile))
  loss = np.mean(shortfall) / np.max(balances - np.min(balances))  # Scale by range
  return loss


def expected_shortfall(level, balances):
  """
  Loss function based on expected shortfall below the level.

  Args:
      level (float): The level to evaluate for liquidity risk.
      balances (np.ndarray): A 1D array of daily balances.

  Returns:
      float: The expected shortfall loss value.
  """


  shortfall = np.maximum(0, level - balances)
  expected_shortfall_value = np.mean(shortfall[shortfall > 0])  # Only consider actual shortfalls
  return expected_shortfall_value / np.max(balances - np.min(balances))  # Scale by range


def cost_sensitive_loss(level, balances, shortfall_cost=1.0, excess_cost=0.1):
  """
  Loss function with costs for shortfalls and excess liquidity.

  Args:
      level (float): The level to evaluate for liquidity risk.
      balances (np.ndarray): A 1D array of daily balances.
      shortfall_cost (float, optional): Cost multiplier for shortfalls (defaults to 1.0).
      excess_cost (float, optional): Cost multiplier for excess liquidity (defaults to 0.1).

  Returns:
      float: The combined cost-sensitive loss value.
  """

  shortfall = np.maximum(0, level - balances)
  excess = np.maximum(0, balances - level)
  loss = np.mean(shortfall_cost * shortfall + excess_cost * excess) / (np.max(balances) - np.min(balances))  # Scale by range
  return loss
def find_optimal_level(balances, loss_function, method ="diff"):
  """
  Finds the level that minimizes the given loss function.

  Args:
      balances (np.ndarray): A 1D array of daily balances.
      loss_function: The loss function to use for optimization.

  Returns:
      float: The optimal level that minimizes liquidity risk.
  """

  initial_guess = np.mean(balances)  # Good initial guess based on average

  lower_bound = np.min(balances)  # Adjust based on your understanding of balances
  upper_bound = np.max(balances)  # Adjust based on your understanding of balances
  #print (f'lower_bound: {lower_bound} upper_bound {upper_bound}')
  bounds = [(lower_bound, upper_bound)]  # Tuple of bounds for a single variable

#   result = minimize(loss_function, initial_guess, args=(balances,),
#                        method='differential_evolution', bounds=bounds)
  if method =="diff":
      result = differential_evolution(lambda x: loss_function(x, balances), bounds)
  else:
      result = minimize(loss_function, x0=initial_guess, args=(balances,),
                              method='SLSQP', bounds=bounds)
  return result.x[0]
# Visualization of results for each iteration
def visualize_pl_results(balances_data, results):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (client, balances) in enumerate(balances_data.items()):
        axs[i].scatter(range(len(balances)), [0]* len(balances), alpha=0.6)
        axs[i].axhline(y=calculate_profit_or_loss(np.mean(balances),balances), color='red', label=f'Mean: {calculate_profit_or_loss(np.mean(balances),balances):.2f}', linestyle='--')
        axs[i].axhline(y=calculate_profit_or_loss(np.median(balances),balances), color='orange', label=f'Median: {calculate_profit_or_loss(np.median(balances),balances):.2f}', linestyle='-.')
        k=2
        for label, optimal_levels in results.items():
            axs[i].axhline(y=calculate_profit_or_loss(optimal_levels[client],balances),color=f"C{k}",linestyle='--', label=f'{label}: {calculate_profit_or_loss(optimal_levels[client],balances):.2f}')
            k +=1
        axs[i].set_title(client)
        axs[i].set_xlabel('Day')
        axs[i].legend()
        axs[i].set_ylabel('Balance' if i == 0 else '')
    plt.suptitle('P&L Based On Optimal Levels ')
    plt.tight_layout()
    plt.show()
def visualize_optimization_results(balances_data, results):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (client, balances) in enumerate(balances_data.items()):
        axs[i].scatter(range(len(balances)), balances, label='Balances', alpha=0.6)
        axs[i].axhline(y=np.mean(balances), color='red', label=f'Mean: {np.mean(balances):.2f}', linestyle='--')
        axs[i].axhline(y=np.median(balances), color='orange', label=f'Median: {np.median(balances):.2f}', linestyle='-.')
        k=2
        for label, optimal_levels in results.items():
            axs[i].axhline(y=optimal_levels[client],color=f"C{k}",linestyle='--', label=f'{label}: {optimal_levels[client]:.2f}')
            k +=1
        axs[i].set_title(client)
        axs[i].set_xlabel('Day')
        axs[i].legend()
        axs[i].set_ylabel('Balance' if i == 0 else '')
    plt.suptitle('Optimal Levels')
    plt.tight_layout()
    plt.show()
# Execution and comparison framework for iterating over loss function versions
def execute_loss_function_iterations(balances_data):
    results = {}
    functions = [combined_loss,combined_loss1, quantile_loss, expected_shortfall, cost_sensitive_loss]
    iteration_labels = ["combined_loss","combined_loss1", "quantile_loss", "expected_shortfall","cost_sensitive_loss"]
    
    for func, label in zip(functions, iteration_labels):
        optimal_levels = {}
        for client, balances in balances_data.items():
            result = find_optimal_level(balances, func)
           
            optimal_levels[client] = result
        results[label] = optimal_levels

    return results

# Apply differential evolution optimization to find the optimal balance level for each client
de_results = execute_loss_function_iterations(client_data)
visualize_optimization_results(client_data, de_results)
visualize_pl_results(client_data, de_results)
# # Visualization of the optimal levels found by differential evolution
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# for i, (client, balances) in enumerate(balances_data.items()):
#     axs[i].scatter(range(len(balances)), balances, label='Balances', alpha=0.6)
#     axs[i].axhline(y=de_results[client], color='g', linestyle='--', label=f'DE Optimal Level: {de_results[client]:.2f}')
#     axs[i].set_title(client)
#     axs[i].set_xlabel('Day')
#     axs[i].legend()
#     axs[i].set_ylabel('Balance' if i == 0 else '')
# plt.suptitle('Optimal Levels Found by Differential Evolution')
# plt.tight_layout()
# plt.show()

de_results  # Display the optimal levels found by differential evolution
