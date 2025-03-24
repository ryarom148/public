import pandas as pd
import duckdb

# Sample data with duplicates and multiple mappings
source = pd.DataFrame({
    'account': ['A001', 'A002', 'A003', 'A003', 'A004', 'A005'],
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

# Method 1: Using duckdb.from_df() - direct query on pandas DataFrames
# This is the most direct and efficient way when working with pandas DataFrames
result_from_df = duckdb.query_df(
    source, 
    mapping, 
    """
    SELECT 
        s.account, 
        s.database, 
        s.value, 
        m.address, 
        m.name
    FROM 
        df1 s 
    LEFT JOIN 
        df2 m 
    ON 
        s.account = m.account 
        AND CONTAINS(LOWER(m.address), LOWER(s.database))
    ORDER BY 
        s.account, s.value
    """
).to_df()

print("DuckDB solution using duckdb.query_df():")
print(result_from_df)

# Method 2: Using duckdb.from_df() with explicit table names
result_named = duckdb.query_df(
    {"source_table": source, "mapping_table": mapping}, 
    """
    SELECT 
        s.account, 
        s.database, 
        s.value, 
        m.address, 
        m.name
    FROM 
        source_table s 
    LEFT JOIN 
        mapping_table m 
    ON 
        s.account = m.account 
        AND CONTAINS(LOWER(m.address), LOWER(s.database))
    ORDER BY 
        s.account, s.value
    """
).to_df()

print("\nDuckDB solution using duckdb.query_df() with named tables:")
print(result_named)

# Method 3: Alternative approach using connect() and register()
con = duckdb.connect(database=':memory:')
con.register('source', source)
con.register('mapping', mapping)

# Solution 1: Using LIKE operator with case-insensitive matching
query_like = """
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source s 
LEFT JOIN 
    mapping m 
ON 
    s.account = m.account 
    AND LOWER(m.address) LIKE '%' || LOWER(s.database) || '%'
ORDER BY 
    s.account, s.value
"""

result_like = con.execute(query_like).fetchdf()
print("DuckDB solution using LIKE operator:")
print(result_like)

# Solution 2: Using CONTAINS function with case-insensitive matching
query_contains = """
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source s 
LEFT JOIN 
    mapping m 
ON 
    s.account = m.account 
    AND CONTAINS(LOWER(m.address), LOWER(s.database))
ORDER BY 
    s.account, s.value
"""

# DuckDB-specific optimizations

# 1. Create indexes for better join performance
con.execute("CREATE INDEX source_account_idx ON source(account)")
con.execute("CREATE INDEX mapping_account_idx ON mapping(account)")

# 2. Use prepared statements for repeated queries
prepared_query = con.prepare("""
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source s 
LEFT JOIN 
    mapping m 
ON 
    s.account = m.account 
    AND CONTAINS(LOWER(m.address), LOWER(s.database))
ORDER BY 
    s.account, s.value
""")

# Execute the prepared statement
result_prepared = prepared_query.execute().fetchdf()
print("\nDuckDB solution using prepared statement:")
print(result_prepared)

# 3. Using materialized views for repeated access
con.execute("""
CREATE OR REPLACE VIEW source_view AS 
SELECT * FROM source
""")

con.execute("""
CREATE OR REPLACE VIEW mapping_view AS 
SELECT *, LOWER(address) as address_lower FROM mapping
""")

# Query using the views with pre-computed lowercase addresses
query_materialized = """
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source_view s 
LEFT JOIN 
    mapping_view m 
ON 
    s.account = m.account 
    AND CONTAINS(m.address_lower, LOWER(s.database))
ORDER BY 
    s.account, s.value
"""

result_materialized = con.execute(query_materialized).fetchdf()
print("\nDuckDB solution using materialized views:")
print(result_materialized)

# 4. Parallel processing with DuckDB
# Enable parallel execution (uses multiple CPU cores)
con.execute("PRAGMA threads=4")

# Create partitioned data for parallel processing
con.execute("""
CREATE OR REPLACE TABLE source_partitioned AS 
SELECT *, (row_number() OVER()) % 4 AS partition_id 
FROM source
""")

# 5. Use DuckDB explain to analyze query performance
explain_output = con.execute("""
EXPLAIN 
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source s 
LEFT JOIN 
    mapping m 
ON 
    s.account = m.account 
    AND CONTAINS(LOWER(m.address), LOWER(s.database))
""").fetchall()

print("\nDuckDB query plan:")
for row in explain_output:
    print(row[0])

# 6. Using DuckDB's optimized string functions
query_optimized = """
SELECT 
    s.account, 
    s.database, 
    s.value, 
    m.address, 
    m.name
FROM 
    source s 
LEFT JOIN 
    mapping m 
ON 
    s.account = m.account 
    AND POSITION(LOWER(s.database) IN LOWER(m.address)) > 0
ORDER BY 
    s.account, s.value
"""

result_optimized = con.execute(query_optimized).fetchdf()
print("\nDuckDB solution using optimized POSITION function:")
print(result_optimized)

# Performance comparison for larger datasets
def performance_comparison():
    # Create larger datasets
    import numpy as np
    np.random.seed(42)
    
    accounts = [f'A{i:03d}' for i in range(1000)]
    databases = [f'DB_{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}' for i in range(1000)]
    
    # Large source dataframe (10,000 rows)
    large_source = pd.DataFrame({
        'account': np.random.choice(accounts, 10000),
        'database': np.random.choice(databases, 10000),
        'value': np.random.randint(100, 1000, 10000)
    })
    
    # Large mapping dataframe (5,000 rows)
    large_mapping = pd.DataFrame({
        'account': np.random.choice(accounts, 5000),
        'address': [f'Address {i} ({db})' for i, db in 
                   enumerate(np.random.choice(databases, 5000))],
        'name': [f'Person {i}' for i in range(5000)]
    })
    
    # Register with DuckDB
    con.register('large_source', large_source)
    con.register('large_mapping', large_mapping)
    
    # Time the DuckDB query
    import time
    
    print("\nPerformance comparison with larger datasets (10,000 source rows, 5,000 mapping rows):")
    
    # DuckDB timing
    start = time.time()
    duck_result = con.execute("""
        SELECT 
            s.account, 
            s.database, 
            s.value, 
            m.address, 
            m.name
        FROM 
            large_source s 
        LEFT JOIN 
            large_mapping m 
        ON 
            s.account = m.account 
            AND CONTAINS(LOWER(m.address), LOWER(s.database))
    """).fetchdf()
    duck_time = time.time() - start
    print(f"DuckDB execution time: {duck_time:.4f} seconds")
    print(f"Result rows: {len(duck_result)}")
    
    # Pandas timing (using the concise approach from earlier)
    source_with_id = large_source.copy().assign(_id=range(len(large_source)))
    
    start = time.time()
    merged = pd.merge(source_with_id, large_mapping, on='account', how='left')
    merged['_match'] = merged.apply(
        lambda row: pd.notna(row['address']) and str(row['database']).lower() in str(row['address']).lower(), 
        axis=1
    )
    matches = merged[merged['_match']]
    non_matches = merged[~merged['_match']].drop_duplicates('_id')
    matched_ids = set(matches['_id'])
    non_matches = non_matches[~non_matches['_id'].isin(matched_ids)]
    pandas_result = pd.concat([matches, non_matches]).sort_values('_id').drop('_match', axis=1).drop('_id', axis=1)
    pandas_time = time.time() - start
    print(f"Pandas execution time: {pandas_time:.4f} seconds")
    print(f"Result rows: {len(pandas_result)}")
    
    print(f"DuckDB is {pandas_time/duck_time:.1f}x faster than Pandas")

import pandas as pd
import duckdb

# Sample data based on your images
source_df = pd.DataFrame({
    'User_Name': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'Database_Name': ['db_prod1', 'db_dev2', 'db_test3', 'cdb_prod1', 'db_unknown'],
    'Value': [100, 200, 300, 400, 500]
})

mapping_df = pd.DataFrame({
    'User Name': ['user1', 'user2', 'user3', 'user4', 'user6'],
    'Account Name': ['acc_prod1', 'acc_dev2', 'acc_test3', 'acc_prod4', 'acc_other'],
    'Address': ['123 Main St (srvc_prod1)', '456 Oak Ave (srvc_dev1)', 
                '789 Pine Rd (srvc_test1)', '101 Palm Blvd (srvc_prod2)', 
                '555 Other Ln (srvc_other)']
})

# Oracle mapping table based on your second image
oracle_mapping = pd.DataFrame({
    'CDB_NAME': ['cdb_prod1', 'cdb_dev1', 'cdb_test1', 'cdb_prod1', 'cdb_dev1', 'cdb_prod1'],
    'SERVICE_NAME': ['srvc_prod1', 'srvc_dev1', 'srvc_test1', 'srvc_prod2', 'srvc_dev2', 'srvc_prod3'],
    'PDB_NAME': ['db_prod1', 'db_dev2', 'db_test3', 'db_prod4', 'db_dev5', 'db_prod6']
})

def merge_step_by_step(source_df, mapping_df, oracle_mapping):
    # Step 1: Join mapping_df with oracle_mapping on Address and SERVICE_NAME
    step1_query = """
    SELECT 
        m.*,
        om.* EXCLUDE (SERVICE_NAME)
    FROM 
        mapping_df m
    LEFT JOIN 
        oracle_mapping om
    ON 
        CONTAINS(LOWER(m.Address), LOWER(om.SERVICE_NAME))
    """
    
    # Execute step 1
    step1_result = duckdb.query_df(
        {"mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        step1_query
    ).to_df()
    
    print("Step 1 - Join mapping_df with oracle_mapping:")
    print(step1_result)
    
    # Step 2: Merge source_df with the result from step 1
    step2_query = """
    WITH enhanced_mapping AS (
        -- This is the result from step 1
        SELECT 
            m.*,
            om.PDB_NAME,
            om.CDB_NAME,
            om.SERVICE_NAME
        FROM 
            mapping_df m
        LEFT JOIN 
            oracle_mapping om
        ON 
            CONTAINS(LOWER(m.Address), LOWER(om.SERVICE_NAME))
    )
    
    SELECT 
        s.*,
        m.* EXCLUDE ("User Name")
    FROM 
        source_df s
    LEFT JOIN 
        enhanced_mapping m
    ON 
        TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
        AND (
            -- Rule 1: Database_Name matches PDB_NAME
            TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.PDB_NAME))
            OR
            -- Rule 2: Database_Name matches CDB_NAME
            TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.CDB_NAME))
            OR
            -- Rule 3: Database_Name exists in Account Name
            CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name)))
            OR
            -- Rule 4: Database_Name exists in Address
            CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name)))
        )
    ORDER BY
        s.User_Name
    """
    
    # Execute step 2
    step2_result = duckdb.query_df(
        {"source_df": source_df, "mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        step2_query
    ).to_df()
    
    print("\nStep 2 - Final merge with the four rules:")
    print(step2_result)
    
    return step2_result

# Execute the step-by-step merge
final_result = merge_step_by_step(source_df, mapping_df, oracle_mapping)

import pandas as pd
import duckdb

# Sample data based on your images
source_df = pd.DataFrame({
    'User_Name': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'Database_Name': ['db_prod1', 'db_dev2', 'db_test3', 'cdb_prod1', 'db_unknown'],
    'Value': [100, 200, 300, 400, 500]
})

mapping_df = pd.DataFrame({
    'User Name': ['user1', 'user2', 'user3', 'user4', 'user6'],
    'Account Name': ['acc_prod1', 'acc_dev2', 'acc_test3', 'acc_prod4', 'acc_other'],
    'Address': ['123 Main St (srvc_prod1)', '456 Oak Ave (srvc_dev1)', 
                '789 Pine Rd (srvc_test1)', '101 Palm Blvd (srvc_prod2)', 
                '555 Other Ln (srvc_other)']
})

# Oracle mapping table based on your second image
oracle_mapping = pd.DataFrame({
    'CDB_NAME': ['cdb_prod1', 'cdb_dev1', 'cdb_test1', 'cdb_prod1', 'cdb_dev1', 'cdb_prod1'],
    'SERVICE_NAME': ['srvc_prod1', 'srvc_dev1', 'srvc_test1', 'srvc_prod2', 'srvc_dev2', 'srvc_prod3'],
    'PDB_NAME': ['db_prod1', 'db_dev2', 'db_test3', 'db_prod4', 'db_dev5', 'db_prod6']
})

def corrected_merge(source_df, mapping_df, oracle_mapping):
    """
    Corrected merge approach that properly handles the four rules
    """
    query = """
    WITH 
    -- First, join mapping with oracle mapping
    mapping_with_oracle AS (
        SELECT 
            m.*,
            om.PDB_NAME,
            om.CDB_NAME,
            om.SERVICE_NAME
        FROM 
            mapping_df m
        LEFT JOIN 
            oracle_mapping om
        ON 
            CONTAINS(LOWER(m.Address), LOWER(om.SERVICE_NAME))
    ),
    
    -- Match source with the combined mapping
    matched_data AS (
        SELECT 
            s.*,
            m.* EXCLUDE ("User Name")
        FROM 
            source_df s
        LEFT JOIN 
            mapping_with_oracle m
        ON 
            TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
            AND (
                -- Rule 1: Database_Name matches PDB_NAME (if PDB_NAME is not null)
                (m.PDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.PDB_NAME)))
                OR
                -- Rule 2: Database_Name matches CDB_NAME (if CDB_NAME is not null)
                (m.CDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.CDB_NAME)))
                OR
                -- Rule 3: Database_Name exists in Account Name
                CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name)))
                OR
                -- Rule 4: Database_Name exists in Address
                CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name)))
            )
    )
    
    SELECT * FROM matched_data
    ORDER BY User_Name
    """
    
    result = duckdb.query_df(
        {"source_df": source_df, "mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        query
    ).to_df()
    
    return result

# Execute the corrected merge
result = corrected_merge(source_df, mapping_df, oracle_mapping)
print("Corrected merge result:")
print(result)

# Alternative approach with more explicit rule prioritization
def merge_with_rule_priority(source_df, mapping_df, oracle_mapping):
    """
    Merge with explicitly prioritized rules
    """
    query = """
    WITH 
    -- Join mapping with oracle mapping
    mapping_with_oracle AS (
        SELECT 
            m.*,
            om.PDB_NAME,
            om.CDB_NAME,
            om.SERVICE_NAME
        FROM 
            mapping_df m
        LEFT JOIN 
            oracle_mapping om
        ON 
            CONTAINS(LOWER(m.Address), LOWER(om.SERVICE_NAME))
    ),
    
    -- Prepare source data with all potential matches
    source_matches AS (
        SELECT 
            s.User_Name,
            s.Database_Name,
            s.Value,
            m."Account Name",
            m.Address,
            m.PDB_NAME,
            m.CDB_NAME,
            m.SERVICE_NAME,
            CASE
                WHEN m.PDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.PDB_NAME)) THEN 1
                WHEN m.CDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.CDB_NAME)) THEN 2
                WHEN CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name))) THEN 3
                WHEN CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name))) THEN 4
                ELSE 0
            END AS match_type
        FROM 
            source_df s
        LEFT JOIN 
            mapping_with_oracle m
        ON 
            TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
        WHERE
            match_type > 0  -- Only keep actual matches
    ),
    
    -- For each source row, select the match with highest priority
    best_matches AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY User_Name, Database_Name, Value
                ORDER BY match_type
            ) AS priority_rank
        FROM source_matches
    )
    
    SELECT * EXCLUDE (match_type, priority_rank)
    FROM best_matches
    WHERE priority_rank = 1
    ORDER BY User_Name
    """
    
    result = duckdb.query_df(
        {"source_df": source_df, "mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        query
    ).to_df()
    
    return result

# Execute the prioritized merge
priority_result = merge_with_rule_priority(source_df, mapping_df, oracle_mapping)
print("\nMerge with rule prioritization:")
print(priority_result)

# Final optimized merge function that correctly handles all rules
def final_merge(source_df, mapping_df, oracle_mapping):
    """
    Final optimized merge function
    """
    query = """
    WITH mapping_enhanced AS (
        -- Pre-join mapping_df with oracle_mapping to create enhanced mapping
        SELECT 
            m.*,
            om.PDB_NAME,
            om.CDB_NAME,
            om.SERVICE_NAME
        FROM 
            mapping_df m
        LEFT JOIN 
            oracle_mapping om
        ON 
            CONTAINS(LOWER(m.Address), LOWER(om.SERVICE_NAME))
    )
    
    SELECT 
        s.*,
        m.* EXCLUDE ("User Name")
    FROM 
        source_df s
    LEFT JOIN 
        mapping_enhanced m
    ON 
        TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
        AND (
            -- Check the four rules independently
            CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name)))
            OR
            CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name)))
            OR
            (m.PDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.PDB_NAME)))
            OR
            (m.CDB_NAME IS NOT NULL AND TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(m.CDB_NAME)))
        )
    ORDER BY
        s.User_Name
    """
    
    return duckdb.query_df(
        {"source_df": source_df, "mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        query
    ).to_df()

# Execute the final optimized merge
final_result = final_merge(source_df, mapping_df, oracle_mapping)
print("\nFinal optimized merge result:")
print(final_result)

import pandas as pd

def merge_with_or_condition(left_df, right_df, left_keys, right_keys, how='inner'):
    """
    Merge two DataFrames based on multiple key pairs using OR logic.
    
    Parameters:
    -----------
    left_df : DataFrame
        Left DataFrame
    right_df : DataFrame
        Right DataFrame
    left_keys : list
        List of columns from left_df to use for matching
    right_keys : list
        List of columns from right_df to use for matching
    how : str
        Type of merge to perform ('inner', 'left', 'right', 'outer')
    
    Returns:
    --------
    DataFrame : Merged result
    """
    # Create individual merges, one for each key pair
    dfs = []
    for left_key, right_key in zip(left_keys, right_keys):
        merged = pd.merge(
            left_df, right_df,
            left_on=left_key, right_on=right_key,
            how=how, suffixes=('', f'_{right_key}'),
            indicator=True
        )
        # Mark which rows matched on this key pair
        merged['_matched_on'] = f"{left_key}={right_key}"
        dfs.append(merged)
    
    # Combine all the individual merges
    combined = pd.concat(dfs)
    
    # Remove duplicates (where a row matched on multiple key pairs)
    # Sort to prioritize certain matches if needed
    result = combined.sort_values('_matched_on').drop_duplicates(
        subset=left_df.columns.tolist(), 
        keep='first'
    )
    
    # Clean up intermediate columns
    result = result.drop(columns=['_merge', '_matched_on'])
    
    return result

# Example usage
left_df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'tom@example.com'],
    'data': ['A', 'B', 'C', 'D', 'E']
})

right_df = pd.DataFrame({
    'user_id': [2, 3, 6, 7, 5],
    'email': ['other@example.com', 'bob@example.com', 'mary@example.com', 'steve@example.com', 'tom@example.com'],
    'value': [100, 200, 300, 400, 500]
})

# Merge where either id=user_id OR email=email
result = merge_with_or_condition(
    left_df, right_df,
    left_keys=['id', 'email'],
    right_keys=['user_id', 'email']
)

print(result)

#########################################
import pandas as pd
import duckdb

def merge_substring_mapping(main_df, mapping_df, 
                           main_column='user_name', 
                           mapping_column='user_id',
                           prefer_longer_match=True):
    """
    Merge two pandas DataFrames where mapping_column values may be substrings
    of main_column values. Handles overlapping substrings carefully.
    
    Parameters:
    -----------
    main_df : DataFrame
        Main DataFrame (left side of the join)
    mapping_df : DataFrame
        Mapping/dictionary DataFrame (right side of the join)
    main_column : str
        Column in main_df that contains the strings to search within
    mapping_column : str
        Column in mapping_df that contains the substrings to look for
    prefer_longer_match : bool
        If True, prioritize longer substring matches when multiple matches exist
        
    Returns:
    --------
    DataFrame : Result of the left join
    """
    # Register DataFrames with DuckDB
    tables = {
        "main_table": main_df,
        "mapping_table": mapping_df
    }
    
    # Optimized SQL query
    query = f"""
    WITH 
    -- Add normalized columns for comparison
    main_normalized AS (
        SELECT 
            *,
            LOWER(TRIM({main_column})) AS normalized_main
        FROM 
            main_table
    ),
    
    mapping_normalized AS (
        SELECT 
            *,
            LOWER(TRIM({mapping_column})) AS normalized_mapping,
            LENGTH(TRIM({mapping_column})) AS substring_length
        FROM 
            mapping_table
    ),
    
    -- Find all possible matches with ranking in a single step
    ranked_matches AS (
        SELECT 
            m.*,
            d.*,
            ROW_NUMBER() OVER (
                PARTITION BY {', '.join([f'm.{col}' for col in main_df.columns])}
                ORDER BY 
                    d.substring_length {'DESC' if prefer_longer_match else 'ASC'},  -- Order by length based on preference
                    d.{mapping_column}  -- Consistent tiebreaker
            ) AS match_rank
        FROM 
            main_normalized m
        CROSS JOIN 
            mapping_normalized d
        WHERE 
            POSITION(d.normalized_mapping IN m.normalized_main) > 0
    )
    
    -- Get only the best match for each main row
    SELECT 
        m.* EXCLUDE (normalized_main),
        r.* EXCLUDE (normalized_main, normalized_mapping, substring_length, match_rank, {', '.join([f'r.{col}' for col in main_df.columns])})
    FROM 
        main_normalized m
    LEFT JOIN 
        ranked_matches r
    ON 
        {' AND '.join([f'm.{col} = r.{col}' for col in main_df.columns])}
        AND r.match_rank = 1
    """
    
    # Execute query
    result = duckdb.query_df(tables, query).to_df()
    
    return result



# Perform the merge, preferring longer substring matches
result = merge_substring_mapping(main_df, mapping_df)
