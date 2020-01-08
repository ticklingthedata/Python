from selenium import webdriver
import hamcrest as h
import pandas as pd
from tabulate import tabulate
import datetime


def assert_value_in_title():

    tested_values = ['Google', 'Bing']
    df_results = pd.DataFrame(columns=['Test', 'Result', 'Details', 'Duration (s)'])

    for tested_value in tested_values:

        assertion_error_raised = False
        driver = webdriver.Firefox()
        test_start = datetime.datetime.now()
        driver.get("https://google.com")

        try:
            h.assert_that(driver.title, h.contains_string(tested_value))
        except AssertionError as assertion_error:
            test_end = datetime.datetime.now()
            assertion_error_raised = True
            df_results = df_results.append({'Test': tested_value + ' in page title', 'Result': 'FAIL', 'Details':
                                            assertion_error, 'Duration (s)': (test_end-test_start).total_seconds()},
                                           ignore_index=True)

        if not assertion_error_raised:
            test_end = datetime.datetime.now()
            df_results = df_results.append({'Test': tested_value + ' in page title', 'Result': 'SUCCESS', 'Details':
                                            'Condition is met for: ' + tested_value, 'Duration (s)': (test_end-test_start).
                                           total_seconds()}, ignore_index=True)

        driver.close()

    return df_results


df_test_results = assert_value_in_title()

print(tabulate(df_test_results, headers='keys', tablefmt='fancy_grid'))
df_test_results.to_csv('ui_automation_results.csv')





