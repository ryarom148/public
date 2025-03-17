import pandas as pd
import numpy as np

# Sample data with duplicates and multiple mappings
# Adding an example of an account that doesn't exist in mapping
source = pd.DataFrame({
    'account': ['A001', 'A002', 'A003', 'A003', 'A004', 'A005'],  # A005 doesn't exist in mapping
    'database': ['DB_NY', 'DB_CA', 'DB_TX', 'DB_TX', 'DB_FL', 'DB_WA'],
    'value': [100, 200, 300, 350, 400, 500]
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

def left_join_with_substring_match(source_df, mapping_df):
    """
    Performs a left join between source and mapping dataframes based on:
    1. Exact match on account column
    2. Substring match of database within address
    
    Preserves all source rows, including those without matches in mapping.
    """
    # Create a unique identifier for each source row
    source_df = source_df.copy()
    source_df['_source_id'] = range(len(source_df))
    
    # Step 1: Left join on account
    merged = pd.merge(
        source_df, 
        mapping_df, 
        on='account', 
        how='left'  # Critical: use left join to keep all source rows
    )
    
    # Step 2: Filter rows where database is in address, but keep NaN addresses
    # Need to handle NaN values in address column
    merged['_match'] = merged.apply(
        lambda row: pd.notna(row['address']) and str(row['database']) in str(row['address']),
        axis=1
    )
    
    # Step 3: For each source row, keep either:
    # - Matching rows where database is in address
    # - One row with NaN for address/name if no matches found
    
    # First, get all matching rows
    matches = merged[merged['_match']]
    
    # Second, for each source row without a match, keep one row with NaNs
    no_matches = merged[
        (~merged['_match']) & 
        (~merged['_source_id'].isin(matches['_source_id']))
    ].drop_duplicates('_source_id')
    
    # Combine matching and non-matching rows
    result = pd.concat([matches, no_matches])
    
    # Sort by original source order and drop temporary columns
    result = result.sort_values('_source_id')
    result = result.drop(['_source_id', '_match'], axis=1)
    
    return result

# Execute the left join with substring matching
result = left_join_with_substring_match(source, mapping)

print("Left join with substring matching:")
print(result)

# For a simplified version that's still efficient:
def simplified_left_join_substring(source_df, mapping_df):
    """A more concise version of the left join with substring matching"""
    # Add source ID
    source_with_id = source_df.copy().reset_index().rename(columns={'index': '_source_id'})
    
    # Group mapping by account for faster processing
    mapping_by_account = {account: group for account, group in mapping_df.groupby('account')}
    
    results = []
    
    # Process each source row
    for _, row in source_with_id.iterrows():
        account = row['account']
        database = row['database']
        
        # Check if account exists in mapping
        if account in mapping_by_account:
            # Find addresses containing the database string
            matching_rows = mapping_by_account[account][
                mapping_by_account[account]['address'].str.contains(database, regex=False)
            ]
            
            if not matching_rows.empty:
                # If matches found, add them to results
                for _, match_row in matching_rows.iterrows():
                    results.append({
                        **row.to_dict(),
                        'address': match_row['address'],
                        'name': match_row['name']
                    })
            else:
                # No substring match found, add row with NaNs
                results.append({
                    **row.to_dict(),
                    'address': np.nan, 
                    'name': np.nan
                })
        else:
            # Account not in mapping, add row with NaNs
            results.append({
                **row.to_dict(),
                'address': np.nan, 
                'name': np.nan
            })
    
    # Convert to DataFrame and sort by original order
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('_source_id')
    result_df = result_df.drop('_source_id', axis=1)
    
    return result_df

# Run the simplified version
simplified_result = simplified_left_join_substring(source, mapping)

print("\nSimplified approach result:")
print(simplified_result)

# Ultra-concise version for those who prefer one-liners
def concise_left_join_substring(source_df, mapping_df):
    """Ultra-concise version of left join with substring matching"""
    # Add source ID and perform left join
    source_with_id = source_df.copy().assign(_id=range(len(source_df)))
    
    # First, perform left join on account
    merged = pd.merge(source_with_id, mapping_df, on='account', how='left')
    
    # Mark rows where database is found in address (handling NaNs)
    merged['_match'] = merged.apply(
        lambda row: pd.notna(row['address']) and row['database'] in str(row['address']), 
        axis=1
    )
    
    # Keep matching rows plus one non-matching row per source row
    matches = merged[merged['_match']]
    non_matches = merged[~merged['_match']].drop_duplicates('_id')
    
    # Get source IDs that have at least one match
    matched_ids = set(matches['_id'])
    
    # Keep only non-matches for source rows that don't have any matches
    non_matches = non_matches[~non_matches['_id'].isin(matched_ids)]
    
    # Combine matches and necessary non-matches, sort, and clean up
    return pd.concat([matches, non_matches]).sort_values('_id').drop('_match', axis=1).drop('_id', axis=1)

# Run the concise version
concise_result = concise_left_join_substring(source, mapping)

print("\nConcise approach result:")
print(concise_result)
