import os
import sys
import types
import asyncio
from CodeInterpreter_async import RobustCodeInterpreter
#from code_interpriter_03 import RobustCodeInterpreter

# Ensure the current directory is in the Python path
sys.path.append(os.getcwd())

# Create a RobustCodeInterpreter instance
interpreter = RobustCodeInterpreter(os.getcwd())

# Test Case 1: Basic module with third-party library
module1 = """
import numpy as np
xren=1000
def calculate_mean(numbers):
    return np.mean(numbers)

result = calculate_mean([1, 2, 3, 4, 5])
print(f"The mean is: {result}")
"""

# Test Case 2: Module importing from module1
module2 = """
from module1 import calculate_mean,xren
import pandas as pd
print(xren)
def process_data(data):
    df = pd.DataFrame(data)
    mean = calculate_mean(df['values'])
    return f"Processed mean: {mean}"

result = process_data({'values': [10, 20, 30, 40, 50]})
print(result)
"""

# Test Case 3: Module with multiple imports and a class
module3 = """

import module2
import module1
from scipy import stats

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def analyze(self):
        mean = module1.calculate_mean(self.data)
        median = stats.skew(self.data)
        mode = stats.mode(self.data).mode[0]
        return f"Mean: {mean}, Median: {median}, Mode: {mode}"

analyzer = DataAnalyzer([1, 2, 2, 3, 4, 5, 5])
print(analyzer.analyze())
"""

# Test Case 4: Module using all previous modules and plotting
module4 = """
import module1
import module2
import module3
import matplotlib.pyplot as plt

def visualize_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title('Data Visualization')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig('data_plot.png')
    plt.close()

    analyzer = module3.DataAnalyzer(data)
    analysis = analyzer.analyze()
    processed = module2.process_data({'values': data})
    
    return f"Visualization saved. {analysis}. {processed}"

result = visualize_data([1, 3, 2, 5, 4, 6, 8, 7])
print(result)
"""

# Run the test cases
# test_cases = [
#     ('module1.py', module1),
#     ('module2.py', module2),
#     ('module3.py', module3),
#     ('module4.py', module4)
# ]

# for filename, code in test_cases:
#     print(f"\nExecuting {filename}:")
#     result = interpreter.execute_code(code, {}, filename)
#     if result['success']:
#         print("Execution successful.")
#         print("Output:", result['output'])
#     else:
#         print("Execution failed.")
#         print("Error:", result['error'])

# Reset the interpreter after all tests
# filename = 'module4.py'
# code = module4
# print(f"\nExecuting {filename}:")
# result = interpreter.execute_code(code, {}, filename)
# if result['success']:
#     print("Execution successful.")
#     print("Output:", result['output'])
# else:
#     print("Execution failed.")
#     print("Error:", result['error'])
#interpreter.reset()

# Add all modules to the interpreter
interpreter.add_module('module4.py', module4)
#interpreter.add_module('module2.py', module2)
interpreter.add_module('module3.py', module3)
interpreter.add_module('module1.py', module1)

#Execute a single module (it will automatically execute dependencies)
results = await interpreter.execute_all()
print("\nExecuting module1:")
result =  await interpreter.execute_module('module2')
if result['success']:
    print("Execution successful.")
    print("Output:", result['output'])
else:
    print("Execution failed.")
    print("Error:", result['error'])

# Or execute all modules in the correct order
print("\nExecuting all modules:")
results = interpreter.execute_all()
for module_name, result in results.items():
    print(f"\nModule: {module_name}")
    if result['success']:
        print("Execution successful.")
        print("Output:", result['output'])
    else:
        print("Execution failed.")
        print("Error:", result['error'])

# # Reset the interpreter after all tests
# interpreter.reset()