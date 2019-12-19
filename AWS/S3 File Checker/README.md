# S3 File Check'n'Plot: 

## Context
Drake s3 files history should be monitored an checked for discrepencies.

## Objectives:
The logic aims to:

1. Automatically check the s3 content for a given time frame
2. Help identifying discrepencies and the mismatches with the expectations.

## Overview of S3 File Check'n'Plot:

1. The script reads the expectation data from the data file.
2. It searches s3 files in a given time frame.
3. It compares the files date to the expectations data.
4. It highlights mismatches and plots some fancy graphs to vizualise results.

## File structure:
1. A root folder containing:
- The main script file.
- The data file. 

## Data file expectations naming rules:
1. One expectation per line
2. Expectation format: bucket,bank,naming convention,expected arrival time,expected path
3. Example: datavault-sangiovese,TD,TD_3PM_FTSE_MM_Quotes,15:03:00,sftp-prod/raw/incoming/FTSE/PROD

## Prerequisites:
1. AWS SSO configuration: https://ticksmith.atlassian.net/wiki/spaces/DevOps/pages/678691136/AWS+-+SSO+-+Configuring+your+credentials 
2. Python 3.7.1 or above.
3. Following Python libraries:
- pandas
- os
- boto3
- arrow
- datefinder
- seaborn
- matplotlib

## Advantages :
1. Eliminates manual file checks.
2. Pinpoints what's wrong with the files.



