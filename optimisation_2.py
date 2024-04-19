import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import boxcox
import scipy.stats as stats
from scipy.optimize import differential_evolution
# def loss_function(level, balances):
#   """
#   Calculates a combined loss function for liquidity risk optimization.

#   Args:
#       level (float): The level to evaluate for liquidity risk.
#       balances (np.ndarray): A 1D array of daily balances.

#   Returns:
#       float: The combined loss value.
#   """

#   n = len(balances)
#   max_diff = np.max(np.abs(balances - np.mean(balances)))  # Max difference for scaling

#   total_difference = np.sum(np.abs(balances - level)) / (n * max_diff)
#   variance_ratio = np.var(balances - level) / max_diff**2
#   below_optimal = np.sum(balances < level) / n
#   changes_across = np.sum(np.abs(np.diff(balances - level > 0))) / (n - 1)

#   loss = total_difference + variance_ratio + below_optimal + changes_across
#   return loss

def loss_function1(level, balances):
    n = len(balances)
    max_diff = np.max(np.abs(balances - np.mean(balances)))  # Max difference for scaling

    total_difference = np.sum(np.abs(balances - level)) / (n * max_diff)
    variance_ratio = np.var(balances - level) / max_diff**2
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
  loss = np.mean(shortfall_cost * shortfall + excess_cost * excess) / np.max(balances - np.min(balances))  # Scale by range
  return loss


def find_optimal_level(balances, loss_function):
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
  print (f'lower_bound: {lower_bound} upper_bound {upper_bound}')
  bounds = [(lower_bound, upper_bound)]  # Tuple of bounds for a single variable

#   result = minimize(loss_function, initial_guess, args=(balances,),
#                        method='differential_evolution', bounds=bounds)
  result = differential_evolution(lambda x: loss_function(x, balances), bounds)
    #de_results[client] = result.x[0]

  optimal_level = result.x[0]
  return optimal_level


def transform_data(balances, transformation="identity"):
  """
  Applies a transformation to the daily balances before optimization.

  Args:
      balances (np.ndarray): A 1D array of daily balances.
      transformation (str, optional): The transformation to apply (defaults to "identity").

  Returns:
      np.ndarray: The transformed balances.
  """

  if transformation == "log":
    return np.log(balances + 1)  # Avoid log(0)
  elif transformation == "coxbox":
    return stats.boxcox(balances + 1)[0]  # Use SciPy's boxcox function
  elif transformation == "sqrt":
    return np.sqrt(balances)
  elif transformation == "power":
    # Consider a power of 0.5 or other values based on analysis
    return balances**0.5
  elif transformation == "identity":
    return balances
  else:
    raise ValueError(f"Unsupported transformation: {transformation}")


def test_data_transformation(balances):
  """
  Analyzes the distribution and suggests a transformation.

  Args:
      balances (np.ndarray): A 1D array of daily balances.

  Returns:
      str: The recommended transformation (e.g., "log", "coxbox", "sqrt", "power").
  """

  # Check for skewness (positive or negative)
  skewness = np.abs(stats.skew(balances))

  # Decision rules for transformation
  if skewness > 0.5:
    if balances.min() > 0:  # Log transformation suitable
      return "log"
    else:  # Consider Box-Cox or square root for non-positive values
      return "coxbox" if np.std(balances) > np.mean(balances) else "sqrt"
  elif skewness < -0.5:  # Negative skewness
    return "power"  # Consider power transformation (e.g., 0.5)
  else:
    return "identity"  # No transformation needed for symmetric distribution


def visualize_optimal_levels(balances, loss_functions, labels, investment_horizon=22):
  """
  Visualizes the optimal levels for each loss function.

  Args:
      balances (np.ndarray): A 1D array of daily balances.
      loss_functions (list): A list of loss functions to use for optimization.
      labels (list): A list of labels for each loss function.
      investment_horizon (int, optional): The investment horizon in days (defaults to 22).
  """


 # Sort loss functions and labels together based on loss function names
  #sorted_loss_functions, sorted_labels = zip(sorted(zip(loss_functions, labels), key=lambda x: x[0].__name__))
   # Sort loss functions and labels together based on loss function names
  sorted_list = sorted(zip(loss_functions, labels), key=lambda x: x[0].__name__)
  sorted_loss_functions, sorted_labels = zip(*sorted_list)  # Unpack using unpacking operator

  num_functions = len(loss_functions)


  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(balances, label="Daily Balances")

  # Decide on transformation based on data distribution
  transformation = test_data_transformation(balances)
  transformed_balances = balances #transform_data(balances, transformation)

  # Find optimal levels for each loss function (using transformed data)
  optimal_levels = [find_optimal_level(transformed_balances, lf) for lf in sorted_loss_functions]

  # Plot transformed balances (optional)
  if transformation != "identity":
    ax.plot(np.arange(len(balances)), transformed_balances, label="Transformed Balances")

  # Plot horizontal lines for optimal levels (transformed back for visualization)
  for i, level in enumerate(optimal_levels):
    optimal_level_for_viz = (np.exp(level) - 1) if transformation == "log" else (
        stats.inv_boxcox(level, (balances + 1).min()) if transformation == "coxbox" else level**2
        if transformation == "sqrt" else level**(1/0.5)  # Raise to power of 0.5 for recommended power transformation
    )
    ax.axhline(
        y=optimal_level_for_viz,
        color=f"C{i}",
        linestyle="--",
        label=f"{sorted_labels[i]} (Optimal after {investment_horizon} days)",
    )

  ax.set_xlabel("Days")
  ax.set_ylabel("Balance" if transformation == "identity" else "Transformed Balance")

  title = f"Optimal Levels for Liquidity Risk (Transformation: {transformation})"
  ax.set_title(title)
  ax.grid(True)
  ax.legend()

  plt.show()


# Example usage
balances = np.array([100, 120, 95, 110, 80, 135, 105, 90, 140, 115, 75, 125, 85, 100, 98, 150, 120, 88, 112, 108, 92, 130])

loss_functions = [loss_function1, quantile_loss, expected_shortfall, cost_sensitive_loss]
labels = ["Combined Loss", "Quantile-Based", "Expected Shortfall", "Cost-Sensitive"]

investment_horizon = 22  # Investment horizon in days

visualize_optimal_levels(balances, loss_functions, labels, investment_horizon)

