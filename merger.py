import pandas as pd

# Sample data with duplicates and multiple mappings
source = pd.DataFrame({
    'account': ['A001', 'A002', 'A003', 'A003', 'A004'],
    'database': ['DB_NY', 'DB_CA', 'DB_TX', 'DB_TX', 'DB_FL'],
    'value': [100, 200, 300, 350, 400]
})

mapping = pd.DataFrame({
    'account': ['A001', 'A001', 'A002', 'A003', 'A003', 'A004'],
    'address': [
        '123 Main St, New York (DB_NY)', 
        '789 Second Ave, New York (DB_NY)',
        '456 Oak Ave, California (DB_CA)', 
        '789 Pine Rd, Texas (DB_TX)',
        '321 Cedar Ln, Texas (DB_TX2)',
        '101 Palm Blvd, Florida (DB_FL)'
    ],
    'name': ['John', 'John', 'Alice', 'Bob', 'Bob', 'Carol']
})

def efficient_merge(source_df, mapping_df):
    """
    Efficiently merge source and mapping dataframes based on account match
    and database substring in address.
    """
    # Add a temporary unique identifier to source
    source_df = source_df.copy()
    source_df['_temp_id'] = range(len(source_df))
    
    # Step 1: Create a cross join setup by adding a key column with constant value
    source_df['_key'] = 1
    mapping_df = mapping_df.copy()
    mapping_df['_key'] = 1
    
    # Step 2: Merge on key and account to get all possible combinations
    # that match on account
    df_merged = pd.merge(
        source_df, 
        mapping_df, 
        on=['_key', 'account'], 
        how='inner'
    )
    
    # Step 3: Filter to keep only rows where database is in address
    # This is a vectorized operation, much faster than apply
    df_merged['_match'] = df_merged['address'].str.contains(
        df_merged['database'], regex=False
    )
    df_result = df_merged[df_merged['_match']]
    
    # Step 4: Sort by original source order and drop temporary columns
    df_result = df_result.sort_values('_temp_id')
    df_result = df_result.drop(['_key', '_temp_id', '_match'], axis=1)
    
    return df_result

# Execute the efficient merge
result = efficient_merge(source, mapping)

print("Elegant and efficient merge result:")
print(result)

# One-liner version for those who prefer concise code
# (Same logic, just more compact)
def concise_merge(source_df, mapping_df):
    """One-liner version of the efficient merge function"""
    source_with_id = source_df.copy().assign(_id=range(len(source_df)))
    return (pd.merge(
        source_with_id, mapping_df, on='account', how='inner'
    ).loc[lambda df: df.apply(
        lambda row: row['database'] in row['address'], axis=1
    )].sort_values('_id').drop('_id', axis=1))

# Execute the concise version
concise_result = concise_merge(source, mapping)

print("\nConcise version result:")
print(concise_result)
