# API automation V3

## Context
All exposed APIs need to be automatically tested in every release. 

## Objectives:
The API automated tests aim to:

1. Automatically test the different exposed APIs as part of the release process.
2. Minimize the effort of the manual and repetitive API testing tasks.
3. Organize and simplify the testing process (writing, execution and debugging).

## Overview of the API automation V3 ecosystem:
This version has been simplified compared to V1 and V2:

1. The testing data:
- A simple csv file  "[api_name]_test_data.csv" containing the base test data.
- Final test data is automatically generated from the test_data file.   

2. The testing logic:
- Every row in the test data file is considered a test case.
- Logic will loop and execute all the tests until no test is left.

## Automation structure:

1. Every API is represented by a folder containing the test logic, the test data and the test results files.
2. Test logic files are .py. Test data and test results are .csv. Names are self explanatory.
3. Automation entry point is [api_name]_main.py.


## Automation structure naming rules:
The are 3 basic rules:

1. The test data file name has to be: [api_name]_test_data.csv.
2. The test names has to be the ones in the provided [api_name]_test_data.csv.
3. The test base endpoints has to be the ones provided in the [api_name]_test_data.csv
(It's possible to change the client in the endpoint).

or simply:

The only thing you have to do is replacing the base endpoint in the file by the base endpoint of your client (and eventually the username/password to use your own user).


## Automation prerequisites:
1. Python 3.6 or above.
2. Libraries in the requirements.txt file.

## Test creation and execution steps:
1. Create the base endpoint for the right test name.
2. Let the automation do the rest.

## Advantages :
1. Minimal test creation steps.
2. Designed to be integrated to release pipelines with minimal human involvement
(feed it with the right base data and let it create the test data, execute it and generate the results).



