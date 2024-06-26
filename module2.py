from module1 import calculate_mean,xren
import pandas as pd
print(xren)
def process_data(data):
    df = pd.DataFrame(data)
    mean = calculate_mean(df['values'])
    return f"Processed mean: {mean}"

result = process_data({'values': [10, 20, 30, 40, 50]})
print(result)