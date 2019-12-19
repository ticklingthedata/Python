# Quik overview:
This tool exists to compare the content of the inventory table and s3 bucket data. This could be a part of any data checking/assesement process.

# How does it work?
1. Connects to a database and excutes a query.
2. Creates a file from the results of the quety (inventory files).
3. Connects to s3 and uses the files created in step to as file search.
4. Returnes results by dataset.

# Inputs:
Following files has to reside in the same folder as the script and will be used as script input data:

1. JRTP_db_connection_data.csv: data used to connect to the db.
2. JRTP_s3_connection_data.csv: data used to connect to s3.
3. JRTP_datasets.csv: dataset list.
5. s3_search_file.txt: auto-generated search file from db query.
6. DB query: user input.

# Outputs:
Following files will be created in the same folder as the script and will be used as script output data:

1. search_report.csv: search results by dataset.
2. inventory_log.csv: detailed results of the inventory to s3 comparison.
3. s3_log.csv: detailed results of the s3 to inventory comparison.

# Requirements:

- Additional required libraries:
  - pandas
  - tabulate
  - progressbar
 
# Changelog

## [2.0] - Mar 25, 2019
### Added
- Reading db and s3 connection data from external csv files.
- Reading search data from the auto-generated search file.
- A fancy progress bar indicating the % completed, elapsed and remaining times.
- A search report showing the results : dataset, S3 Count, Inventory Count, S3 Count/Inventory Count %, Inventory Count/S3 Count %, Missing from S3 Count, Missing from Inventory Count.



