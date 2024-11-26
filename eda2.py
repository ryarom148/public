import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Set global plotting style
sns.set(style='whitegrid', context='talk', palette='deep')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

GROUP_COLUMNS = ['client_id', 'location_id']
DATE_COLUMN = 'date'
YEAR_COLUMN = 'year'
MONTH_COLUMN = 'month'
BALANCE_COLUMN = 'daily_balance'
def prepare_data(df):
    """
    Prepare the DataFrame by ensuring date columns are datetime, sorting the data,
    and adding 'year' and 'month' columns.
    """
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.sort_values(by=DATE_COLUMN, inplace=True)
    df['year'] = df[DATE_COLUMN].dt.year
    df['month'] = df[DATE_COLUMN].dt.month
    return df

# def identify_top_clients(df):
#     """
#     Identify the subset of clients whose combined average balance equals the sum of the remaining clients.
#     Returns the list of top clients, their quantile, and the dataframe of average balances.
#     """
#     # Calculate average balance per client-location
#     df_avg = df.groupby(GROUP_COLUMNS)[BALANCE_COLUMN].mean().reset_index()
#     df_avg.columns = GROUP_COLUMNS + ['avg_balance']

#     # Sort clients by average balance in descending order
#     df_avg.sort_values(by='avg_balance', ascending=False, inplace=True)

#     # Calculate cumulative sum and total sum
#     df_avg['cumulative_sum'] = df_avg['avg_balance'].cumsum()
#     total_balance = df_avg['avg_balance'].sum()
#     df_avg['cumulative_perc'] = df_avg['cumulative_sum'] / total_balance

#     # Find the point where cumulative balance equals half of the total balance
#     half_total_balance = total_balance / 2
#     df_avg['is_top_client'] = df_avg['cumulative_sum'] <= half_total_balance

#     top_clients = df_avg[df_avg['is_top_client']][GROUP_COLUMNS]
#     quantile = len(top_clients) / len(df_avg)

#     return top_clients, quantile, df_avg

# def plot_top_clients_pie(df_avg, quantile):
#     """
#     Create a pie chart displaying the composition and absolute balances (aggregated)
#     between top clients and others.
#     """
#     # Aggregate balances
#     total_balance = df_avg['avg_balance'].sum()
#     top_balance = df_avg[df_avg['is_top_client']]['avg_balance'].sum()
#     other_balance = total_balance - top_balance

#     # Prepare data for pie chart
#     labels = ['Top Clients', 'Other Clients']
#     sizes = [top_balance, other_balance]
#     explode = (0.1, 0)  # Explode the top clients slice for emphasis

#     # Create pie chart
#     fig, ax = plt.subplots()
#     ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#            startangle=90, colors=['#ff9999', '#66b3ff'])
#     ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
#     plt.title('Composition of Total Average Balance')
#     plt.tight_layout()
#     plt.show()

#     print(f"Top clients represent {quantile*100:.2f}% of all client-location pairs.")
#     print(f"Top Clients Total Average Balance: {top_balance:.2f}")
#     print(f"Other Clients Total Average Balance: {other_balance:.2f}")
def identify_top_clients(df, target_cumulative_perc=50):
    """
    Identify the subset of clients whose combined average balance equals the target cumulative percentage.

    Parameters:
        df (pd.DataFrame): The prepared DataFrame.
        target_cumulative_perc (float): The target cumulative percentage (e.g., 50 for 50%).

    Returns:
        top_clients (pd.DataFrame): DataFrame of top clients.
        quantile (float): The percentile of top clients.
        df_avg (pd.DataFrame): DataFrame containing average balances of all clients.
    """
    # Calculate average balance per client-location
    df_avg = df.groupby(GROUP_COLUMNS)[BALANCE_COLUMN].mean().reset_index()
    df_avg.columns = GROUP_COLUMNS + ['avg_balance']

    # Sort clients by average balance in descending order
    df_avg.sort_values(by='avg_balance', ascending=False, inplace=True)

    # Calculate cumulative sum and total sum
    df_avg['cumulative_sum'] = df_avg['avg_balance'].cumsum()
    total_balance = df_avg['avg_balance'].sum()
    df_avg['cumulative_perc'] = df_avg['cumulative_sum'] / total_balance * 100

    # Identify top clients where cumulative sum <= target percentage
    df_avg['is_top_client'] = df_avg['cumulative_sum'] <= (target_cumulative_perc / 100) * total_balance

    # Select top clients
    top_clients = df_avg[df_avg['is_top_client']][GROUP_COLUMNS]
    quantile = len(top_clients) / len(df_avg)

    return top_clients, quantile, df_avg

def plot_top_clients_pie(df_avg, quantile):
    """
    Create a pie chart displaying the composition of top clients vs others based on number of clients.

    Parameters:
        df_avg (pd.DataFrame): DataFrame containing average balances of all clients.
        quantile (float): The percentile of top clients.
    """
    # Calculate number of clients in each group
    num_top_clients = df_avg[df_avg['is_top_client']].shape[0]
    num_other_clients = df_avg.shape[0] - num_top_clients

    # Calculate total balance for each group
    top_balance = df_avg[df_avg['is_top_client']]['avg_balance'].sum()
    other_balance = df_avg['avg_balance'].sum() - top_balance

    # Convert to millions
    top_balance_m = top_balance / 1e6
    other_balance_m = other_balance / 1e6

    # Prepare data for pie chart
    labels = ['Top Clients', 'Other Clients']
    sizes = [num_top_clients, num_other_clients]
    explode = (0.1, 0)  # Explode the top clients slice for emphasis

    # Calculate percentages
    total_clients = num_top_clients + num_other_clients
    percentages = [num_top_clients / total_clients * 100, num_other_clients / total_clients * 100]

    # Create pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels,
        autopct=lambda pct: f"{int(pct/100 * total_clients)} ({pct:.1f}%)",
        startangle=90, colors=['#ff9999', '#66b3ff'],
        pctdistance=0.85, textprops={'fontsize': 12}
    )

    # Draw circle for a donut chart effect
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Composition of Clients by Average Balance Contribution', fontsize=18)
    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"Top clients represent {quantile*100:.2f}% of all client-location pairs.")
    print(f"Top Clients Total Average Balance: ${top_balance_m:.2f}M")
    print(f"Other Clients Total Average Balance: ${other_balance_m:.2f}M")


def plot_balance_distribution(df_avg):
    """
    Plot the distribution of aggregated average balances across the company based on quantiles.

    Parameters:
        df_avg (pd.DataFrame): DataFrame containing average balances of all clients.
    """
    # Define quantiles
    quantiles = [5, 25, 50, 75, 95]
    quantile_values = np.percentile(df_avg['avg_balance'], quantiles)

    # Assign each client to a quantile
    df_avg['balance_quantile'] = pd.cut(
        df_avg['avg_balance'],
        bins=[-np.inf] + list(quantile_values) + [np.inf],
        labels=[f'{quantiles[i]}th' for i in range(len(quantiles))] + ['95th+']
    )

    # Aggregate data for each quantile
    distribution = df_avg.groupby('balance_quantile').agg(
        num_clients=('avg_balance', 'count'),
        aggregated_balance=('avg_balance', 'sum')
    ).reset_index()

    # Calculate actual quantile labels for X-axis
    quantile_labels = ['5th', '25th', '50th', '75th', '95th', '95th+']

    # Plot
    plt.figure(figsize=(14, 7))
    sns.barplot(data=distribution, x='balance_quantile', y='num_clients', palette='viridis')

    # Annotate data labels with aggregated balances in millions
    for index, row in distribution.iterrows():
        plt.text(index, row['num_clients'] + max(distribution['num_clients']) * 0.01,
                 f"${row['aggregated_balance']/1e6:.1f}M",
                 ha='center', va='bottom', fontsize=12)

    plt.title('Distribution of Aggregated Average Balances Across the Company', fontsize=18)
    plt.xlabel('Balance Quantile')
    plt.ylabel('Number of Clients')
    plt.tight_layout()
    plt.show()

    # Description
    print("The bar chart illustrates the distribution of aggregated average balances across different balance quantiles.")
    print("Each bar represents a balance quantile (5th, 25th, 50th, 75th, 95th, and 95th+), showing the number of clients within each quantile.")
    print("Data labels indicate the total aggregated average balance for each quantile in millions of dollars.")
def calculate_distribution_quantiles(df_avg, quantiles=[5, 25, 50, 75, 95]):
    """
    Calculate the distribution of aggregated average balances based on cumulative sum.

    Parameters:
        df_avg (pd.DataFrame): DataFrame containing average balances of all clients.
        quantiles (list): List of quantiles to calculate.

    Returns:
        distribution (pd.DataFrame): DataFrame containing quantile, number of clients, and aggregated balances.
    """
    # Sort the DataFrame by average balance ascending
    df_sorted = df_avg.sort_values(by='avg_balance', ascending=True).reset_index(drop=True)

    # Calculate cumulative sum
    df_sorted['cumulative_sum'] = df_sorted['avg_balance'].cumsum()

    # Total balance
    total_balance = df_sorted['avg_balance'].sum()

    # Calculate cumulative percentage
    df_sorted['cumulative_perc'] = df_sorted['cumulative_sum'] / total_balance * 100

    # Define the quantile thresholds
    quantile_thresholds = quantiles

    # Initialize list to store distribution data
    distribution_data = []

    previous_threshold = 0
    previous_clients = 0
    previous_sum = 0

    for threshold in quantile_thresholds:
        # Find the index where cumulative_perc >= threshold
        idx = df_sorted[df_sorted['cumulative_perc'] >= threshold].index.min()

        if np.isnan(idx):
            idx = len(df_sorted) - 1

        # Number of clients in this quantile
        num_clients = idx - previous_clients

        # Aggregated balance in this quantile
        aggregated_balance = df_sorted.loc[previous_clients:idx-1, 'avg_balance'].sum()

        distribution_data.append({
            'quantile': f"{threshold}th",
            'num_clients': num_clients,
            'aggregated_balance_m': aggregated_balance / 1e6  # Convert to millions
        })

        # Update previous thresholds
        previous_threshold = threshold
        previous_clients = idx

    # Handle the last segment (95th+)
    if quantiles[-1] < 100:
        num_clients = len(df_sorted) - previous_clients
        aggregated_balance = df_sorted.loc[previous_clients:, 'avg_balance'].sum()
        distribution_data.append({
            'quantile': f"{quantiles[-1]}th+",
            'num_clients': num_clients,
            'aggregated_balance_m': aggregated_balance / 1e6
        })

    distribution = pd.DataFrame(distribution_data)

    return distribution

def plot_balance_distribution1(distribution):
    """
    Plot the distribution of aggregated average balances across the company based on quantiles.

    Parameters:
        distribution (pd.DataFrame): DataFrame containing quantile, number of clients, and aggregated balances.
    """
    plt.figure(figsize=(14, 7))
    sns.barplot(data=distribution, x='quantile', y='num_clients', palette='viridis')

    # Annotate data labels with aggregated balances in millions
    for index, row in distribution.iterrows():
        plt.text(index, row['num_clients'] + max(distribution['num_clients']) * 0.01,
                 f"${row['aggregated_balance_m']:.1f}M",
                 ha='center', va='bottom', fontsize=12)

    plt.title('Distribution of Aggregated Average Balances Across the Company', fontsize=18)
    plt.xlabel('Balance Quantile')
    plt.ylabel('Number of Clients')
    plt.tight_layout()
    plt.show()

    # Description
    print("The bar chart illustrates the distribution of aggregated average balances across different balance quantiles.")
    print("Each bar represents a balance quantile (5th, 25th, 50th, 75th, 95th, and 95th+), showing the number of clients within each quantile.")
    print("Data labels indicate the total aggregated average balance for each quantile in millions of dollars.")

def calculate_operating_balances_per_client_location(df, group_columns, date_column, balance_column, year_column, month_column):
    df[date_column] = pd.to_datetime(df[date_column])
    daily_sum = df.groupby(group_columns + [date_column]).agg(
        daily_balance=(balance_column, 'sum')
    ).reset_index()
    daily_sum[year_column] = daily_sum[date_column].dt.year
    daily_sum[month_column] = daily_sum[date_column].dt.month
    operating_balances = daily_sum.groupby(group_columns + [year_column, month_column])[balance_column].agg(
        mean_balance='mean',
        median_balance='median',
        quantile_5=lambda x: x.quantile(0.05)
    ).reset_index()
    return operating_balances, daily_sum

def aggregate_operating_balances(operating_balances, year_column, month_column):
    bank_operating_balances = operating_balances.groupby([year_column, month_column]).agg(
        mean_balance_bank=('mean_balance', 'sum'),
        median_balance_bank=('median_balance', 'sum'),
        quantile_5_bank=('quantile_5', 'sum')
    ).reset_index()
    bank_operating_balances[year_column] = bank_operating_balances[year_column].astype(int)
    bank_operating_balances[month_column] = bank_operating_balances[month_column].astype(int)
    return bank_operating_balances

def aggregate_and_plot_operating_balances(operating_balances, daily_sum, year, group_name='Bank'):
    bank_operating_balances = aggregate_operating_balances(operating_balances,YEAR_COLUMN,MONTH_COLUMN)
    operating_year = bank_operating_balances[bank_operating_balances[YEAR_COLUMN] == year]
    daily_sum_year = daily_sum[daily_sum[YEAR_COLUMN] == year]
    bank_daily_sum = daily_sum_year.groupby(DATE_COLUMN)['daily_balance'].sum().reset_index()
    months_colors = sns.color_palette("hsv", 12)
    month_color_dict = {month: color for month, color in zip(range(1, 13), months_colors)}
    fig, ax = plt.subplots(figsize=(16,8))
    for _, row in bank_daily_sum.iterrows():
        current_date = row[DATE_COLUMN]
        current_month = current_date.month
        ax.vlines(current_date, ymin=0, ymax=row['daily_balance'],
                  color=month_color_dict[current_month], linewidth=0.5)
    for _, row in operating_year.iterrows():
        year_val = int(row[YEAR_COLUMN])
        month_val = int(row[MONTH_COLUMN])
        month_start = pd.Timestamp(year=year_val, month=month_val, day=1)
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        if month_val == 1:
            ax.hlines(y=row['mean_balance_bank'], xmin=month_start, xmax=month_end,
                      color='blue', linestyle='--', label='Mean Balance')
            ax.hlines(y=row['median_balance_bank'], xmin=month_start, xmax=month_end,
                      color='orange', linestyle='-.', label='Median Balance')
            ax.hlines(y=row['quantile_5_bank'], xmin=month_start, xmax=month_end,
                      color='green', linestyle=':', label='5th Percentile Balance')
        else:
            ax.hlines(y=row['mean_balance_bank'], xmin=month_start, xmax=month_end,
                      color='blue', linestyle='--')
            ax.hlines(y=row['median_balance_bank'], xmin=month_start, xmax=month_end,
                      color='orange', linestyle='-.')
            ax.hlines(y=row['quantile_5_bank'], xmin=month_start, xmax=month_end,
                      color='green', linestyle=':')
        ax.axvline(x=month_end, color='black', linestyle=':', linewidth=1)
    months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='MS')
    for month in months:
        ax.text(month + pd.Timedelta(days=15), ax.get_ylim()[1]*0.95, month.strftime('%b'),
                rotation=0, fontsize=10, ha='center', va='top')
    ax.set_title(f'Daily Sum of Balances with Operating Measures - {group_name} - {year}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Daily Balance')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    plt.show()


def calculate_errors(operating_balances, daily_sum, group_columns, date_column, year_column, month_column, balance_column, operating_balance_columns):
    merged = daily_sum.merge(operating_balances, on=group_columns + [year_column, month_column], how='left')
    for measure, op_balance_col in operating_balance_columns.items():
        merged[f'error_{measure}'] = merged[balance_column] - merged[op_balance_col]
    error_melt = merged.melt(id_vars=[date_column, year_column, month_column], 
                             value_vars=[f'error_{measure}' for measure in operating_balance_columns.keys()],
                             var_name='measure', value_name='error')
    error_melt['measure'] = error_melt['measure'].str.replace('error_', '').str.capitalize()
    errors = error_melt.groupby(['measure', date_column]).agg(
        sum_negative_errors=('error', lambda x: x[x < 0].sum()),
        sum_positive_errors=('error', lambda x: x[x > 0].sum()),
        total_error=('error', 'sum')
    ).reset_index()
    errors['year'] = errors[date_column].dt.year
    return errors

def plot_operating_errors(errors, measure, year, sum_neg_col='sum_negative_errors', 
                          sum_pos_col='sum_positive_errors', total_err_col='total_error', 
                          date_column='date', group_name='Bank'):
    months_colors = sns.color_palette("hsv", 12)
    month_color_dict = {month: color for month, color in zip(range(1, 13), months_colors)}
    errors_filtered = errors[(errors['measure'] == measure) & (errors['year'] == year)]
    errors_daily = errors_filtered.groupby(date_column).agg(
        sum_negative_errors=(sum_neg_col, 'sum'),
        sum_positive_errors=(sum_pos_col, 'sum'),
        total_error=(total_err_col, 'sum')
    ).reset_index()
    NEGATIVE_COLOR = 'red'
    POSITIVE_COLOR = 'green'
    fig, ax1 = plt.subplots(figsize=(16,8))
    for _, row in errors_daily.iterrows():
        current_date = row[date_column]
        current_month = current_date.month
        ax1.vlines(current_date, ymin=row['sum_negative_errors'], ymax=0,
                  color=NEGATIVE_COLOR, linewidth=0.5) #month_color_dict[current_month]
        ax1.vlines(current_date, ymin=0, ymax=row['sum_positive_errors'], #month_color_dict[current_month]
                  color=POSITIVE_COLOR, linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(errors_daily[date_column], errors_daily['total_error'], color='black', linestyle='--', label='Total Error')
    months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='MS')
    for month in months:
        ax1.axvline(x=month, color='black', linestyle=':', linewidth=1)
    for month in months:
        ax1.text(month + pd.Timedelta(days=15), ax1.get_ylim()[1]*0.95, month.strftime('%b'),
                 rotation=0, fontsize=10, ha='center', va='top')
     # Format y-axis labels to display in millions with thousands separator
    formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:,.1f}M')
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sum of Negative and Positive Errors')
    ax2.set_ylabel('Total Error')
    ax1.set_title(f'Errors Over Time - {measure} - {group_name} - {year}')
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Sum Negative Errors'),
        Line2D([0], [0], color='orange', lw=2, label='Sum Positive Errors'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Total Error')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()
def plot_total_errors_per_measure(errors, measures, year, date_column='date', total_err_col='total_error', group_name='Bank'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    sns.set(style='whitegrid', context='talk', palette='deep')
    plt.rcParams['figure.figsize'] = (16, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    colors = sns.color_palette("tab10", n_colors=len(measures))
    fig, ax = plt.subplots(figsize=(16,8))

    for measure, color in zip(measures, colors):
        errors_filtered = errors[(errors['measure'] == measure) & (errors['year'] == year)]
        ax.plot(errors_filtered[date_column], errors_filtered[total_err_col], label=f'{measure} Total Error', color=color)

    formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:,.1f}M')
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel('Date')
    ax.set_ylabel('Total Error (Millions)')
    ax.set_title(f'Total Errors Over Time - {group_name} - {year}')

    ax.legend()

    # Generate statistics
    for measure in measures:
        errors_filtered = errors[(errors['measure'] == measure) & (errors['year'] == year)]
        negative_days = (errors_filtered[total_err_col] < 0).sum()
        positive_days = (errors_filtered[total_err_col] > 0).sum()
        negative_days_ave = (errors_filtered[errors_filtered[total_err_col] < 0])[total_err_col].mean()
        positive_days_ave = (errors_filtered[errors_filtered[total_err_col] > 0])[total_err_col].mean()
        print(f"{measure} - Days Borrowed: {negative_days}, Days Invested: {positive_days}")
        print(f"{measure} - Average Daily Borrowing: {negative_days_ave}, Average Daily Investing: {positive_days_ave}")
    plt.tight_layout()
    plt.show()
def main():
    dates = pd.date_range(start='2021-01-01', periods=252, freq='B')
    client_ids = [f'Client_{i}' for i in range(1, 101)]
    location_ids = [f'Location_{i}' for i in range(1, 6)]
    data = []
    np.random.seed(42)
    for date in dates:
        for client in client_ids:
            for location in location_ids:
                balance = np.random.normal(loc=10000, scale=2000)
                balance = max(balance, 0)
                data.append([client, location, date, balance])
    df = pd.DataFrame(data, columns=['client_id', 'location_id', 'date', 'daily_balance'])

    # Task 1: Identify Top Clients
    top_clients, quantile, df_avg = identify_top_clients(df)
    plot_top_clients_pie(df_avg, quantile)
    # Task 2: Plot Distribution of Aggregated Average Balances
    plot_balance_distribution(df_avg)

    # Task 3: Plot Distribution of Aggregated Average Balances
    distribution = calculate_distribution_quantiles(df_avg)
    plot_balance_distribution1(distribution)
    # Divide data into top clients and others
    df['is_top_client'] = df[GROUP_COLUMNS].apply(tuple, axis=1).isin(top_clients.apply(tuple, axis=1))

    df_top_clients = df[df['is_top_client']]
    df_others = df[~df['is_top_client']]
    operating_balances, daily_sum = calculate_operating_balances_per_client_location(
        df, GROUP_COLUMNS, DATE_COLUMN, BALANCE_COLUMN, YEAR_COLUMN, MONTH_COLUMN)
    bank_operating_balances = aggregate_operating_balances(
        operating_balances, YEAR_COLUMN, MONTH_COLUMN)
    aggregate_and_plot_operating_balances(operating_balances, daily_sum, year=2021, group_name='Bank')
    operating_balance_columns = {'mean': 'mean_balance', 'median': 'median_balance', 'quantile_5': 'quantile_5'}
    errors = calculate_errors(
        operating_balances, daily_sum, GROUP_COLUMNS, DATE_COLUMN, YEAR_COLUMN, MONTH_COLUMN, BALANCE_COLUMN, operating_balance_columns)
    plot_operating_errors(errors, measure='Mean', year=2021, group_name='Bank')
    measures = ['Mean', 'Median', 'Quantile_5']
    plot_total_errors_per_measure(errors, measures, year=2021, group_name='Bank')

if __name__ == "__main__":
    main()