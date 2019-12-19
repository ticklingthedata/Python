# SFTP automation tests

## Context
Used by clients to download/upload files using SFTP, and being manually tested before the release. 

## Objectives:
SFTP automated tests aim to:

1. Automatically test the download/upload of the files using SFTP.
2. Minimize the effort of the manual and repetitive SFTP testing tasks.
3. Organize and simplify the testing process (writing, execution and debugging).

## Overview of SFTP test automation ecosystem:
The SFTP automation test ecosystem can be described as follows:

1. The testing data:
- sftp_connection_data.csv: data file used by the logic as input data for the tests.
- sftp_log.txt: this file contains the logged messages during the test session.
- downloaded files: these are the files downloaded during the tests and that will be immediately deleted after test termination.

2. The testing logic:
- Testing logic files are: sftp_consumer.py 
- Logic can be described as below:
    - the script reads the test data and creates a dataframe to store that data.
    - It will the loop on every row untill the end of the dataframe.
    - The script randomizes the file to download/upload to make the process as near to real usage as possible.
    - The script checks that the file has been correctly downloaded/uploaded (by comparing the remote and the local file sizes)
    - At the end of the tests the script exists with the number of failures.

## Automation structure:
1. A root folder containing the testing data folders and the tesing logic files.

## Automation prerequisites:
1. A user with a private key file in the sftp automation folder to access the remote test/uat servers (info should be modified in the connection data csv file).
2. Python 3.7.1 or above.
3. Following Python libraries:
  - paramiko
  - warnings
  - pandas
  - os
  - stat
  - sys
  - random
  - time

## Test creation and execution steps:
1. Create the data files needed for your test.
2. Place the files created in step 1 in the same folder as the script file.
3. Execute sftp_consumer.py
4. Check the log for additional information about test execution.

## Benefits :
1. Data and logic layers decoupled: changes in one of them do not affect the other.
3. Production ready: the solution does not need considerable configuration/adaptation efforts.
4. Simplified testing: create your test data and let the automation logic do the rest.
