import MySQLdb
import pandas as pd
from tabulate import tabulate as tab
import boto3
import csv
import os
import progressbar


def set_db_connection_data():

    print('\033[4mAVAILABLE ENVIRONMENTS:\033[0m')

    print('\n')

    df_connection = pd.read_csv('JRTP_db_connection_data.csv')

    for index in range(len(df_connection)):
        print(index, '-', df_connection.loc[index]['client'])

    print('\n')

    selected_client = int(input('SELECT THE ENVIRONMENT TO CONNECT TO: '))
    db_host = df_connection.iloc[selected_client]['endpoint']
    db_user = df_connection.iloc[selected_client]['username']
    db_password = df_connection.iloc[selected_client]['password']
    db_instance = df_connection.iloc[selected_client]['database']

    return db_host, db_user, db_password, db_instance


def connect_to_db(db_host, db_user, db_password, db_instance):

    print('\n')
    db = MySQLdb.connect(db_host, db_user, db_password, db_instance)
    print('\033[4mDATABASE CONNECTION STATUS:\033[0m')

    if db.get_host_info() != '':
        print('\n')
        print('SUCCESSFULLY CONNECTED TO', db_instance, 'DATABASE', 'AS', db_user, '.')

    print('\n')

    db_cursor = db.cursor()

    return db_cursor


def execute_query(db_cursor):

    print('\033[4mPROVIDE QUERY AND DOUBLE-PRESS ENTER WHEN FINISHED:\033[0m', '\n')

    raw_query = list()
    while True:
        line = input()
        if line:
            raw_query.append(line)
        else:
            break
    db_query = ('\n'.join(raw_query))
    db_cursor.execute(db_query)

    return db_cursor


def fetch_query_results(db_cursor):

    cursor_data = db_cursor.fetchall()
    raw_list = [list(i) for i in cursor_data]
    db_inventory_files_list = list()

    for raw_list_counter in range(len(raw_list)):
        next_url = ((str(raw_list[raw_list_counter]).strip("[")).strip("]")).strip("'")
        db_inventory_files_list.append(next_url)

    print('QUERY RETURNED ', len(db_inventory_files_list), ' FILES.')
    print('\n')

    return db_inventory_files_list


def write_s3_search_file(db_inventory_files_list):

    with open('s3_search_file.txt', 'w') as s3_search_file:
        for db_inventory_file in db_inventory_files_list:
            s3_search_file.write(db_inventory_file + '\n')


def set_s3_connection_data():

    print('\033[4mAVAILABLE S3 PROFILES:\033[0m')
    print('\n')

    df_s3 = pd.read_csv('JRTP_s3_connection_data.csv')

    for index in range(len(df_s3)):
        print(index, '-', df_s3.loc[index]['profile'])

    print('\n')

    selected_profile = int(input('SELECT THE S3 PROFILE: '))
    s3_profile = df_s3.iloc[selected_profile]['profile']
    s3_proxy = df_s3.iloc[selected_profile]['proxy']

    return s3_profile, s3_proxy


def connect_to_s3(s3_profile, s3_proxy):

    os.environ["HTTPS_PROXY"] = s3_proxy
    s3_session = boto3.Session(profile_name=s3_profile)

    print('\n')

    print('\033[4mS3 CONNECTION STATUS:\033[0m')
    if s3_session.profile_name != '':
        print('\n')
        print('SUCCESSFULLY CONNECTED TO S3 AS', s3_session.profile_name, '.')

    print('\n')

    s3_resource = s3_session.resource('s3')

    return s3_resource


def create_path_list():

    path_list = list()
    sliced_path_list = list()

    # Creates a list of paths
    with open('s3_search_file.txt', 'r') as s3_search_file:
        for line in s3_search_file:
            temp_line = line.strip('\n')
            path_list.append(temp_line)

    path_list_index = 0

    # Splits the paths into lists to prepare bucket and prefix assignment
    while path_list_index in range(len(path_list)):
        sliced_path_list.append(str(path_list[path_list_index].split('/')))
        path_list_index += 1

    return path_list, sliced_path_list


def s3_file_search(sliced_path_list, s3_resource):

    print('PERFORMING S3 SEARCH. PLEASE WAIT...')
    print('\n')

    prefix = ''
    raw_prefix = ''
    sliced_path_list_index = 0
    total_count = 0
    s3_files_list = list()
    bar = progressbar.ProgressBar(maxval=len(sliced_path_list)).start()

    while sliced_path_list_index in range(len(sliced_path_list)):
        bar.update(sliced_path_list_index+1)
        inner_path_list = (sliced_path_list[sliced_path_list_index][1:-1]).split(',')

        # Extracts the bucket and the path
        for inner_path_list_index in range(len(inner_path_list)):

            if inner_path_list_index == 0:
                bucket = s3_resource.Bucket(str(inner_path_list[inner_path_list_index])[1:-1])

            else:
                raw_prefix = raw_prefix + str((inner_path_list[inner_path_list_index] + '/').strip())
                prefix = (raw_prefix.replace("'", ""))[:-1]

        for obj in bucket.objects.filter(Prefix=prefix):
            total_count += 1
            next_url = str(obj.Object().bucket_name).strip() + '/' + str(obj.Object().key).strip()
            s3_files_list.append(next_url)

        raw_prefix = ''
        sliced_path_list_index += 1

    bar.finish()
    return s3_files_list


def write_search_results(source_list, destination_list, source_file, source, destination, log_file, print_answer):

    source_counter = 0
    found = False
    not_found_count = 0

    if print_answer == 'Y' or print_answer == 'y':
        print('\n')

        print('WRITING RESULTS INTO', log_file, 'THIS OPERATION MAY TAKE SOME (CRAZY) TIME...')
        with open(log_file, 'w', newline='') as source_log:
            source_log_writer = csv.writer(source_log, delimiter=',', quotechar='"',
                                           quoting=csv.QUOTE_MINIMAL)
            source_log_writer.writerow([source_file, 'FOUND IN ' + destination])

    while source_counter in range(len(source_list)):

        for destination_counter in range(len(destination_list)):

            if destination_list[destination_counter] == source_list[source_counter]:
                found = True
                if print_answer == 'Y' or print_answer == 'y':
                    with open(log_file, 'a', newline='') as inventory_log:
                        inventory_log_writer = csv.writer(inventory_log, delimiter=',', quotechar='"',
                                                          quoting=csv.QUOTE_MINIMAL)
                        inventory_log_writer.writerow([source_list[source_counter], 'Yes'])
                break

        if not found:
            not_found_count += 1
            if print_answer == 'Y' or print_answer == 'y':
                with open(log_file, 'a', newline='') as source_log:
                    source_log_writer = csv.writer(source_log, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                    source_log_writer.writerow([source_list[source_counter], 'No'])

        source_counter += 1
        found = False

    print('\n')

    if print_answer == 'Y' or print_answer == 'y':
        print('FILE WRITING FINISHED. CHECK ', log_file, ' IN YOUR WORKSPACE.')


def create_dataset_list():

    dataset_list = list()
    df_dataset = pd.read_csv('JRTP_datasets.csv')

    for row_counter in range(len(df_dataset)):
        dataset_list.append(df_dataset.loc[row_counter]['dataset'])

    return dataset_list


def create_df_report(dataset_list):

    df_report = pd.DataFrame(columns=['Dataset'])

    for dataset in dataset_list:
        df_report = df_report.append({"Dataset": dataset}, ignore_index=True)

    return df_report


def write_file_counts(file_list, dataset_list, which_count):

    df_count = pd.DataFrame(columns=[which_count])
    row_index = 0

    for dataset in dataset_list:

        match = 0

        for file in file_list:
            if str(dataset) in str(file):
                match += 1

        df_count = df_count.append({which_count: match}, ignore_index=True)
        row_index += 1

    return df_count


def calculate_ratios(df_report, source, destination):

    df_ratios_column = source + '/' + destination + ' %'
    df_ratios = pd.DataFrame(columns=[df_ratios_column])

    for row_index in range(len(df_report)):

        if df_report.loc[row_index][destination] != 0:
            df_ratios = df_ratios.append({df_ratios_column: (df_report.loc[row_index][source]/df_report.loc
                                         [row_index][destination])*float(100)}, ignore_index=True)

        else:
            if df_report.loc[row_index][destination] == df_report.loc[row_index][source]:
                df_ratios = df_ratios.append({df_ratios_column: float(100)}, ignore_index=True)
            else:
                df_ratios = df_ratios.append({df_ratios_column: str(df_report.loc[row_index][source])+' new files'},
                                             ignore_index=True)

    return df_ratios


def calculate_missing_files(df_report, source, destination):

    missing_files_column = 'Missing from ' + source
    df_missing_files = pd.DataFrame(columns=[missing_files_column])

    for row_index in range(len(df_report)):

        if df_report.loc[row_index][source] <= df_report.loc[row_index][destination]:
            df_missing_files = df_missing_files.append({missing_files_column: (df_report.loc[row_index][destination] -
                                                                               df_report.loc[row_index][source])},
                                                       ignore_index=True)

    return df_missing_files


def main_logic():

    host, user, password, instance = set_db_connection_data()
    cursor_connect = connect_to_db(host, user, password, instance)
    cursor_execute = execute_query(cursor_connect)
    inventory_files_list = fetch_query_results(cursor_execute)
    write_s3_search_file(inventory_files_list)
    profile, proxy = set_s3_connection_data()
    session = connect_to_s3(profile, proxy)
    raw_path_list, sliced_paths = create_path_list()
    s3_found_files = s3_file_search(sliced_paths, session)
    datasets = create_dataset_list()
    report = create_df_report(datasets)
    df_s3_count = write_file_counts(s3_found_files, datasets, 'S3 Count')
    report = pd.concat([report, df_s3_count], axis=1)
    df_inventory_count = write_file_counts(raw_path_list, datasets, 'Inventory Count')
    report = pd.concat([report, df_inventory_count], axis=1)
    df_s3_to_inventory_ratio = calculate_ratios(report, 'S3 Count', 'Inventory Count')
    report = pd.concat([report, df_s3_to_inventory_ratio], axis=1)
    df_inventory_to_s3_ratio = calculate_ratios(report, 'Inventory Count', 'S3 Count')
    report = pd.concat([report, df_inventory_to_s3_ratio], axis=1)
    df_s3_missing_files = calculate_missing_files(report, 'S3 Count', 'Inventory Count')
    report = pd.concat([report, df_s3_missing_files], axis=1)
    df_inventory_missing_files = calculate_missing_files(report, 'Inventory Count', 'S3 Count')
    report = pd.concat([report, df_inventory_missing_files], axis=1)
    report.to_csv('search_report.csv')

    print('\n')
    print('\033[4mSEARCH RESULTS BELOW. CHECK search_report.csv IN YOUR WORKSPACE:\033[0m')
    print('\n')
    print(tab(report, headers=report.columns))
    print('\n')

    answer = input('SAVE THE INVENTORY TABLE FULL SEARCH LOG TO DISK? Y/N: ')
    write_search_results(s3_found_files, inventory_files_list, 'INVENTORY FILE', 'INVENTORY', 'S3', 'inventory_log.csv',
                         answer)
    print('\n')

    answer = input('SAVE THE S3 FULL SEARCH LOG TO DISK? Y/N: ')
    write_search_results(inventory_files_list, s3_found_files, 'S3 FILE', 'S3', 'INVENTORY', 's3_log.csv', answer)


main_logic()

