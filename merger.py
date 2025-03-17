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

# Register DataFrames with DuckDB
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
