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

# Uncomment to run performance comparison
# performance_comparison()

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
    'Address': ['123 Main St (db_prod1)', '456 Oak Ave (db_dev2)', 
                '789 Pine Rd (db_test3)', '101 Palm Blvd (cdb_prod1)', 
                '555 Other Ln (db_other)']
})

# Oracle mapping table based on your second image
oracle_mapping = pd.DataFrame({
    'CDB_NAME': ['cdb_prod1', 'cdb_dev1', 'cdb_test1', 'cdb_prod1', 'cdb_dev1', 'cdb_prod1'],
    'SERVICE_NAME': ['srvc_prod1', 'srvc_dev1', 'srvc_test1', 'srvc_prod2', 'srvc_dev2', 'srvc_prod3'],
    'PDB_NAME': ['db_prod1', 'db_dev2', 'db_test3', 'db_prod4', 'db_dev5', 'db_prod6']
})

def merge_duckdb3(source_df, mapping_df, oracle_mapping):
    """
    Enhanced merge function that integrates the Oracle mapping table
    """
    # Register DataFrames with DuckDB
    conn = duckdb.connect(':memory:')
    conn.register('source_df', source_df)
    conn.register('mapping_df', mapping_df)
    conn.register('oracle_mapping', oracle_mapping)
    
    # The enhanced query with Oracle mapping integration
    query = """
    WITH 
    -- First, attempt to match source.Database_Name to PDB_NAME
    pdb_matches AS (
        SELECT 
            s.*,
            om.SERVICE_NAME,
            om.CDB_NAME
        FROM 
            source_df s
        LEFT JOIN 
            oracle_mapping om
        ON 
            TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(om.PDB_NAME))
    ),
    
    -- For records where PDB match failed, try matching to CDB_NAME
    all_matches AS (
        SELECT 
            s.*,
            CASE 
                WHEN s.SERVICE_NAME IS NOT NULL THEN s.SERVICE_NAME
                ELSE om.SERVICE_NAME
            END AS final_service_name,
            CASE 
                WHEN s.CDB_NAME IS NOT NULL THEN s.CDB_NAME
                ELSE om.CDB_NAME
            END AS final_cdb_name
        FROM 
            pdb_matches s
        LEFT JOIN 
            oracle_mapping om
        ON 
            s.SERVICE_NAME IS NULL AND
            TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(om.CDB_NAME))
    ),
    
    -- Perform the final merge with mapping_df
    final_merge AS (
        SELECT 
            s.*,
            m.* EXCLUDE ('User Name')
        FROM 
            all_matches s
        LEFT JOIN 
            mapping_df m
        ON 
            TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
            AND (
                -- Match on SERVICE_NAME in address
                (s.final_service_name IS NOT NULL AND 
                 CONTAINS(LOWER(m.Address), LOWER(s.final_service_name)))
                OR
                -- Match on Database_Name in Account Name or Address
                CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name)))
                OR
                CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name)))
            )
    )
    
    SELECT * FROM final_merge
    ORDER BY User_Name;
    """
    
    result = conn.execute(query).fetchdf()
    return result

# Execute the enhanced merge
result = merge_duckdb3(source_df, mapping_df, oracle_mapping)
print("Enhanced merge result with Oracle mapping:")
print(result)

# Alternative approach with a more streamlined query
def merge_duckdb_streamlined(source_df, mapping_df, oracle_mapping):
    """
    Streamlined version using a direct integration
    """
    query = """
    WITH service_name_lookup AS (
        -- First try to match on PDB_NAME
        SELECT 
            s.User_Name,
            s.Database_Name,
            s.Value,
            COALESCE(
                -- Get SERVICE_NAME from PDB match
                (SELECT STRING_AGG(om.SERVICE_NAME, ',') 
                 FROM oracle_mapping om 
                 WHERE TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(om.PDB_NAME))),
                
                -- If no PDB match, try CDB match
                (SELECT STRING_AGG(om.SERVICE_NAME, ',') 
                 FROM oracle_mapping om 
                 WHERE TRIM(LOWER(s.Database_Name)) = TRIM(LOWER(om.CDB_NAME))),
                
                -- If neither, use original Database_Name
                s.Database_Name
            ) AS lookup_keys
        FROM
            source_df s
    )
    
    SELECT 
        s.*,
        m.* EXCLUDE ("User Name")
    FROM 
        service_name_lookup s
    LEFT JOIN 
        mapping_df m
    ON 
        TRIM(LOWER(s.User_Name)) = TRIM(LOWER(m."User Name"))
        AND (
            -- Try to match any of the lookup keys in address
            (s.lookup_keys <> s.Database_Name AND
             (SELECT COUNT(*) FROM (
                 SELECT UNNEST(STRING_SPLIT(s.lookup_keys, ',')) AS service
              ) subq
              WHERE CONTAINS(LOWER(m.Address), TRIM(LOWER(service)))
             ) > 0)
            OR
            -- If no service match, try direct Database_Name match
            CONTAINS(LOWER(m."Account Name"), TRIM(LOWER(s.Database_Name)))
            OR
            CONTAINS(LOWER(m.Address), TRIM(LOWER(s.Database_Name)))
        )
    ORDER BY
        s.User_Name
    """
    
    return duckdb.query_df(
        {"source_df": source_df, "mapping_df": mapping_df, "oracle_mapping": oracle_mapping},
        query
    ).to_df()

# Execute the streamlined version
streamlined_result = merge_duckdb_streamlined(source_df, mapping_df, oracle_mapping)
print("\nStreamlined merge result:")
print(streamlined_result)
