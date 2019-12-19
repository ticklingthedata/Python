import pandas as pd
import os
import boto3
import arrow
import datefinder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

"""
pandas: dataframes to store and create data outputs.
os: proxy usage (could be not necessary).
boto3: s3 related search.
arrow & datefinder: dates related operations.
matplotlib and seaborn: plotting. 
"""


def create_data_list(data_file_path):

    """
    Creates a list of dictionaries from the datafile.
    This list is the set of expectations that will be used as a reference when comparing \
    files from s3 against expected buckets, naming conventions and paths.
    """

    data_dict = {'bucket': '', 'bank': '', 'naming_convention': '', 'expected_path': ''}
    data_list = list()

    with open(data_file_path, 'r') as data_file:

        for line in data_file:

            raw_line = line.strip('\n')
            raw_line = raw_line.replace(" ", "")
            buffer = ''
            dict_locator = 0
            string_locator = 1
            end_of_line = len(raw_line) - 1

            if string_locator < end_of_line:

                for c in raw_line:

                    if c != ',':
                        buffer = buffer + c

                    else:
                        if dict_locator == 0:
                                data_dict['bucket'] = buffer
                        if dict_locator == 1:
                                data_dict['bank'] = buffer
                        if dict_locator == 2:
                                data_dict['naming_convention'] = buffer

                        buffer = ''
                        dict_locator += 1

                else:
                    data_dict['expected_path'] = buffer

                string_locator += 1

            data_list.append(data_dict)
            data_dict = {'bucket': '', 'bank': '', 'naming_convention': '', 'expected_path': ''}

        return data_list


def create_df_results():

    """
    Creates the empty dataframe of results.
    Will be partially populated by the search_s3_files function.
    Is the final form of the csv output.
    """

    df_results = pd.DataFrame(columns=['File Name', 'Bank', 'Bucket', 'Naming Matches Convention',
                                       'Expected Arrival Date', 'Actual Arrival Date', 'Time Difference in Days',
                                       'Expected Path', 'Actual Path', 'Paths Match', 'Duplicate',
                                       'Duplicate(s) Paths', 'Warnings'])

    return df_results


def create_df_expectations(data_list):

    """
    Creates the dataframe of expectations.
    Populates the expectations dataframe with the output of create_data_list function.
    """

    data_list_index = 0
    df_expectations = pd.DataFrame(columns=['Bucket', 'Bank', 'Naming Convention', 'Expected Path'])

    while data_list_index in range(len(data_list)):

        df_expectations = df_expectations.append({'Bucket': data_list[data_list_index]['bucket'],
                                                  'Bank': data_list[data_list_index]['bank'],
                                                  'Naming Convention': data_list[data_list_index]['naming_convention'],
                                                  'Expected Path': data_list[data_list_index]['expected_path']},
                                                 ignore_index=True)
        data_list_index += 1

    return df_expectations


def connect_to_s3():

    """
    Connects to s3 and gets the session used in search_s3_files function.
    """

    os.environ["HTTPS_PROXY"] = "proxy"
    profile = 'profile'
    session = boto3.Session(profile_name=profile)
    if session.profile_name != '':
        print('\n')
        print('SUCCESSFULLY CONNECTED TO S3 AS', session.profile_name, '.')
        print('\n')
    s3_session = session.resource('s3')

    return s3_session


def select_bucket():

    """
    Gets the targeted buckets from user input.
    Selected bucket will be used in search_s3_files function.
    """

    print('BUCKET LIST: ')
    print('\n')
    print('0 - all buckets')
    print('1 - bank1')
    print('2 - bank2')
    print('3 - bank2')
    print('4 - bank3')
    print('5 - bank4')
    print('6 - bank5')
    print('\n')

    bucket_list = list()
    choice = False

    while not choice:
        bucket_number = input('SELECT A BUCKET FOR HISTORICAL ANALYSIS : ')

        if bucket_number == '0':
            bucket_list = ['bank1', 'bank2', 'bank3', 'bank4',
                           'bank5', 'bank6']
            break
        elif bucket_number == '1':
            bucket_list.append('bank1')
            break
        elif bucket_number == '2':
            bucket_list.append('bank2')
            break
        elif bucket_number == '3':
            bucket_list.append('bank3')
            break
        elif bucket_number == '4':
            bucket_list.append('bank4')
            break
        elif bucket_number == '5':
            bucket_list.append('bank5')
            break
        elif bucket_number == '6':
            bucket_list.append('bank6')
            break
        else:
            print('\n')
            print('CHOICE NOT RECOGNIZED!')
            print('\n')

            choice = False

    return bucket_list


def search_s3_files(bucket_list, s3_session, df_results):

    """
    Finds s3 files given search parameters
    Writes the results to the output dataframe
    """

    start_date = input('INSERT START DATE IN YYYY-MM-DD FORMAT: ')
    print('\n')
    end_date = input('INSERT END DATE IN YYYY-MM-DD FORMAT: ')
    print('\n')

    start_date = arrow.get(start_date)
    end_date = arrow.get(end_date)

    file_name = ''

    for bucket in bucket_list:

        s3_bucket = s3_session.Bucket(bucket)
        for obj in s3_bucket.objects.filter():

            last_modified_date = arrow.get(obj.Object().last_modified).to('local')

            if (last_modified_date >= start_date) and (last_modified_date <= end_date):

                bucket_name = str(obj.Object().bucket_name).strip()
                raw_path = str(obj.Object().key).strip()
                for c in reversed(raw_path):
                    if c != '/':
                        file_name += c
                    else:
                        break
                file_name = file_name[::-1]
                file_path = (raw_path.replace(file_name, ''))[:-1]

                df_results = df_results.append({'Bucket': bucket_name, 'File Name': file_name, 'Actual Path': file_path,
                                                'Actual Arrival Date': last_modified_date.format('YYYY-MM-DD')},
                                               ignore_index=True)
                file_name = ''

    return df_results


def write_bank(df_results):

    """
    Writes the bank names in the output dataframe
    A simple bucket name to bank name mapping avoiding duplication
    """

    for df_results_row_index in range(len(df_results)):

        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank1':
            df_results.loc[df_results_row_index]['Bank'] = 'b1'
        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank2':
            df_results.loc[df_results_row_index]['Bank'] = 'b2'
        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank3':
            df_results.loc[df_results_row_index]['Bank'] = 'b3'
        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank4':
            df_results.loc[df_results_row_index]['Bank'] = 'b4
        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank5':
            df_results.loc[df_results_row_index]['Bank'] = 'b5'
        if df_results.iloc[df_results_row_index]['Bucket'] == 'bank6':
            df_results.loc[df_results_row_index]['Bank'] = 'b6'

    return df_results


def write_expected_arrival_date(df_results):

    """
    Guesses the expected arrival date from the s3 file name.
    If no educated guess, writes None.
    """

    for df_results_row_index in range(len(df_results)):

        current_file = df_results.iloc[df_results_row_index]['File Name']
        found_dates = list(datefinder.find_dates(current_file))

        if len(found_dates) != 0:
            found_date = found_dates[-1]
            parsed_date = arrow.get(found_date).format('YYYY-MM-DD')

        df_results.loc[df_results_row_index]['Expected Arrival Date'] = parsed_date
        parsed_date = None

    return df_results


def write_expected_path(df_results, df_expectations):

    """
    Writes the expected path if s3 file meets a naming convention.
    If not, writes None.
    """

    match = False
    for df_results_row_index in range(len(df_results)):

        for df_expectations_row_index in range(len(df_expectations)):

            if str(df_expectations.iloc[df_expectations_row_index]['Naming Convention']) in \
                    str(df_results.iloc[df_results_row_index]['File Name']):

                df_results.loc[df_results_row_index]['Expected Path'] = \
                    df_expectations.loc[df_expectations_row_index]['Expected Path']
                match = True
                break

        if not match:
            df_results.loc[df_results_row_index]['Expected Path'] = None

    return df_results


def check_naming_convention(df_results, df_expectations):

    """
    Tests s3 file against all naming conventions.
    Writes Yes is a match is found, and No if not.
    """

    for df_results_row_index in range(len(df_results)):

        match = False
        for df_expectations_row_index in range(len(df_expectations)):

            if str(df_expectations.iloc[df_expectations_row_index]['Naming Convention']) in \
                    str(df_results.iloc[df_results_row_index]['File Name']):

                df_results.loc[df_results_row_index]['Naming Matches Convention'] = 'Yes'
                match = True
                break

        if not match:
            df_results.loc[df_results_row_index]['Naming Matches Convention'] = 'No'

    return df_results


def check_time_difference(df_results):

    """
    Calculates the delta between s3 file real arrival date and expected arrival date.
    Time delta is expressed in dates.
    Time delta = 0 is good. Else is bad.
    """

    for df_results_row_index in range(len(df_results)):

        if df_results.iloc[df_results_row_index]['Expected Arrival Date'] is not None:

            raw_time_difference = arrow.get(df_results.iloc[df_results_row_index]['Actual Arrival Date']) \
                              - arrow.get(df_results.iloc[df_results_row_index]['Expected Arrival Date'])
            time_difference = raw_time_difference.days
            df_results.loc[df_results_row_index]['Time Difference in Days'] = time_difference

    return df_results


def check_paths(df_results):

    """
    Compares s3 file real path and expected path.
    Writes Yes is a match is found, and No if not.
    """

    for df_results_row_index in range(len(df_results)):

        if df_results.iloc[df_results_row_index]['Expected Path'] == \
                df_results.iloc[df_results_row_index]['Actual Path']:

            df_results.loc[df_results_row_index]['Paths Match'] = 'Yes'

        else:
            df_results.loc[df_results_row_index]['Paths Match'] = 'No'

    return df_results


def check_duplicates(df_results):

    """
    Checks is s3 file has duplicates.
    Writes Yes is a match is found and s3 file path of the copie(s) in Duplicate Paths column.
    Else, writes No.
    """

    for df_results_row_outer_index in range(len(df_results)):

        duplicate = False
        first = True
        for df_results_row_inner_index in range(len(df_results)):

            if df_results.iloc[df_results_row_inner_index]['File Name'] == \
                    df_results.iloc[df_results_row_outer_index]['File Name'] and \
                    df_results_row_outer_index != df_results_row_inner_index:

                df_results.iloc[df_results_row_outer_index]['Duplicate'] = 'Yes'
                duplicate = True

                if not first:
                    df_results.iloc[df_results_row_outer_index]['Duplicate(s) Paths'] = \
                        df_results.iloc[df_results_row_outer_index]['Duplicate(s) Paths'] + \
                        '\n' + df_results.iloc[df_results_row_inner_index]['Actual Path']
                else:
                    df_results.iloc[df_results_row_outer_index]['Duplicate(s) Paths'] = \
                        df_results.iloc[df_results_row_inner_index]['Actual Path']
                    first = False
        if not duplicate:
            df_results.iloc[df_results_row_outer_index]['Duplicate'] = 'No'

    return df_results


def write_warnings(df_results):

    """
    For every s3 file found, this populates the Warning column with a wrap up of the results.
    If any warning is worth writing, it does write it. Else, it writes No issues found.
    """

    for df_results_row_index in range(len(df_results)):

        first = True
        if df_results.iloc[df_results_row_index]['Naming Matches Convention'] == 'No':
            if first:
                df_results.iloc[df_results_row_index]['Warnings'] = 'File name do not meet any naming convention'
                first = False
            else:
                df_results.iloc[df_results_row_index]['Warnings'] = df_results.iloc[df_results_row_index]['Warnings'] \
                                                                    + '\n' + \
                                                                    'File name do not meet any naming convention'

        if df_results.iloc[df_results_row_index]['Time Difference in Days'] != 0:
            if first:
                df_results.iloc[df_results_row_index]['Warnings'] = 'File was not received in the expected day'
                first = False
            else:
                df_results.iloc[df_results_row_index]['Warnings'] = df_results.iloc[df_results_row_index]['Warnings'] \
                                                                    + '\n' + 'File was not received in the expected day'

        if df_results.iloc[df_results_row_index]['Paths Match'] == 'No':
            if first:
                df_results.iloc[df_results_row_index]['Warnings'] = 'File was not received in the expected path'
                first = False
            else:
                df_results.iloc[df_results_row_index]['Warnings'] = df_results.iloc[df_results_row_index]['Warnings'] \
                                                                    + '\n' + \
                                                                    'File was not received in the expected path'

        if df_results.iloc[df_results_row_index]['Duplicate'] == 'Yes':
            if first:
                df_results.iloc[df_results_row_index]['Warnings'] = 'Check duplicate paths column for file copies'
                first = False
            else:
                df_results.iloc[df_results_row_index]['Warnings'] = df_results.iloc[df_results_row_index]['Warnings'] \
                                                                    + '\n' + \
                                                                    'Check duplicate paths column for file copies'
        if first:
            df_results.iloc[df_results_row_index]['Warnings'] = 'No issues found'

    return df_results


def create_df_metrics():

    """
    Creates a dataframe from the output to be used in the plotting.
    """

    output = 'output.csv'
    df_metrics = pd.read_csv(output, usecols=['Bank', 'Naming Matches Convention', 'Time Difference in Days',
                                              'Paths Match', 'Duplicate'])

    return df_metrics


def plot_results_per_bank(df_metrics):

    """
    Plots some useful graphics for every bank.
    People love graphs, so...
    """

    data_per_bank = ['Naming Matches Convention', 'Time Difference in Days', 'Paths Match', 'Duplicate']

    for data_point in data_per_bank:

        sns.set(style="darkgrid")
        sns.set(rc={'figure.figsize': (16, 9)})
        bar_plot = sns.countplot(x="Bank", hue=data_point, data=df_metrics)
        for p in bar_plot.patches:
            bar_plot.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        bar_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()


def logic_flow():

    """
    Main workflow.
    Step-by-step comments below.
    """

    plot_choice_list = ['y', 'Y', 'n', 'N']
    data_file = 'data_file.txt'
    output = 'output.csv'

    # Creates the parsed data from the data file
    parsed_data_list = create_data_list(data_file)

    # Creates the expectations dataframe from the parsed data
    expectations = create_df_expectations(parsed_data_list)

    # Creates the parsed bucket list using the user input choice
    parsed_bucket_list = select_bucket()

    # Creates an s3 sesssion
    open_s3_session = connect_to_s3()

    # Creates an empty dataframe to store the s3 search results
    df_s3_results = create_df_results()

    # Writes the search results to the previously created dataframe
    results = search_s3_files(parsed_bucket_list, open_s3_session, df_s3_results)

    # Appends the bank names to the results dataframe
    results_bank = write_bank(results)

    # Appends the expected arrival dates to the results dataframe
    results_expected_arrival_date = write_expected_arrival_date(results_bank)

    # Appends the expected paths to the results dataframe
    results_expected_path = write_expected_path(results_expected_arrival_date, expectations)

    # Checks s3 files against naming conventions
    results_naming_convention = check_naming_convention(results_expected_path, expectations)

    # Calculates difference between real and expected arrival dates
    results_time_difference = check_time_difference(results_naming_convention)

    # Compares the real and the expected s3 file paths
    results_check_paths = check_paths(results_time_difference)

    # Checks if any duplicates
    results_check_duplicates = check_duplicates(results_check_paths)

    # Writes warnings if any
    results_warnings = write_warnings(results_check_duplicates)

    # Fills missing values with None
    results_final = results_warnings.fillna('None')

    # Writes to csv output file
    results_final.to_csv(output)

    print('RESULTS WERE WRITTEN THE OUTPUT FILE IN THE SCRIPT FOLDER')
    print('\n')

    # Asks if plotting is needed and does if yes
    plot_choice = input('DO YOU WANT TO PLOT THE RESULTS? Y/N: ')
    while plot_choice not in plot_choice_list:
        print('CHOICE NOT RECOGNIZED!')
        print('\n')
        plot_choice = input('DO YOU WANT TO PLOT THE RESULTS? Y/N: ')

    if plot_choice == 'y' or plot_choice == 'Y':
        metrics_df = create_df_metrics()
        plot_results_per_bank(metrics_df)


# Main call
logic_flow()
