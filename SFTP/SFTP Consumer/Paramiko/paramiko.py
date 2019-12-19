import paramiko as sftp
import warnings
import pandas as pd
import os
import stat
import sys
import datetime
from random import randint
import pysnooper
warnings.filterwarnings(action='ignore', module='.*paramiko.*')

password = pd.read_csv('paramiko_1_test_data.csv').iloc[0]['password']


def read_sftp_data(sftp_data_file):

    df_sftp_data = pd.read_csv(sftp_data_file)
    df_sftp_data.columns = ['client', 'endpoint', 'username', 'password', 'port', 'key', 'type', 'expected',
                            'result', 'filename']
    df_sftp_data = df_sftp_data.fillna('')

    return df_sftp_data


def create_df_test_data(df_sftp_data):

    df_test_data = pd.DataFrame(columns=['client', 'endpoint', 'username', 'password', 'port', 'key', 'type', 'expected',
                                         'result'])

    for row_index in range(len(df_sftp_data)):

        df_test_data = df_test_data.append({'client': df_sftp_data.iloc[row_index]['client'],
                                            'endpoint': df_sftp_data.iloc[row_index]['endpoint'],
                                            'username': df_sftp_data.iloc[row_index]['username'],
                                            'password': df_sftp_data.iloc[row_index]['password'],
                                            'port': df_sftp_data.iloc[row_index]['port'],
                                            'key': df_sftp_data.iloc[row_index]['key'],
                                            'type': df_sftp_data.iloc[row_index]['type'],
                                            'expected': df_sftp_data.iloc[row_index]['expected'],
                                            'result': df_sftp_data.iloc[row_index]['result'],
                                            'filename': df_sftp_data.iloc[row_index]['filename']
                                            }, ignore_index=True)

    return df_test_data


def create_connection_data(df_test_data, row_index):

    test_client = df_test_data.iloc[row_index]['client']
    test_endpoint = df_test_data.iloc[row_index]['endpoint']
    test_username = df_test_data.iloc[row_index]['username']
    test_port = int(df_test_data.iloc[row_index]['port'])
    test_key = df_test_data.iloc[row_index]['key']
    test_type = df_test_data.iloc[row_index]['type']
    test_expected = df_test_data.iloc[row_index]['expected']
    test_result = df_test_data.iloc[row_index]['result']
    test_filename = df_test_data.iloc[row_index]['filename']

    return test_client, test_endpoint, test_username, test_port, test_key, test_type, test_expected,\
        test_result, test_filename


def handler(title, instructions, fields):

    global password

    if len(fields) > 1:
        raise sftp.SSHException("Expecting one field only.")

    return [password]


def connect_to_sftp(test_endpoint, test_port, test_username):

    sftp.util.log_to_file("sftp_log", level="DEBUG")

    transport = sftp.Transport((test_endpoint, test_port),
                               default_max_packet_size=10000, default_window_size=10000)
    transport.connect(username=test_username)
    transport.auth_interactive(test_username, handler)

    sftp_client_from_transport = sftp.SFTPClient.from_transport(transport)

    return sftp_client_from_transport, transport


def get_remote_file_path(sftp_client_from_transport, path):

    sftp.util.log_to_file("sftp_log", level="DEBUG")

    file_attributes = sftp_client_from_transport.lstat(path)

    if stat.S_ISDIR(file_attributes.st_mode):
        path_type = 'folder'
    if stat.S_ISREG(file_attributes.st_mode):
        path_type = 'file'

    return path_type


def create_file_path(sftp_client_from_transport):

    remote_file_path = '.'
    sftp.util.log_to_file("sftp_log", level="DEBUG")

    while get_remote_file_path(sftp_client_from_transport, remote_file_path) == 'folder':
        remote_file_path += '/' + sftp_client_from_transport.listdir(remote_file_path) \
        [randint(0, len(sftp_client_from_transport.listdir(remote_file_path))-1)]

    return remote_file_path


def get_remote_file_size(sftp_client_from_transport, remote_file_path):

    sftp.util.log_to_file("sftp_log", level="DEBUG")

    remote_file_attributes = sftp_client_from_transport.stat(remote_file_path)
    remote_file_size = remote_file_attributes.st_size

    return remote_file_size


def print_totals(transferred, to_be_transferred):
    print("Transferred : {0} mb \tOut of: {1} mb. Completed: {2}%".format(round(transferred/(1024*1024), 2),
                                                                          round(to_be_transferred/(1024*1024), 2),
                                                                          round((transferred/to_be_transferred)*100), 1)
          )


def download_single_file(sftp_client_from_transport, remote_file_path, filename):

    sftp.util.log_to_file("sftp_log", level="DEBUG")

    local_file_path = os.path.dirname(__file__) + '/' + filename
    sftp_client_from_transport.get(remote_file_path, local_file_path, callback=print_totals)

    return local_file_path


def compare_file_size(remote_file_size, local_file_path, failures):

    local_file_attributes = os.stat(local_file_path)
    local_file_size = local_file_attributes.st_size

    if local_file_size != remote_file_size:
        failures += 1

    return failures


def retrieve_local_files():

    local_files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(os.getcwd()):
        for file in f:
            if 'single_file_download' in file:
                local_files.append(os.path.join(r, file))

    return local_files


def clean_local_files(local_files):
    for f in local_files:
        os.remove(f)
        print(f, ' was successfully deleted')


# authentication handler requires that

def main_logic():

    ddl_times = list()
    ddl_file_sizes = list()
    ddl_file_paths = list()

    global password

    failures_count = 0
    sftp_data = read_sftp_data('paramiko_test_data.csv')
    test_data = create_df_test_data(sftp_data)
    test_data_index = 0

    while test_data_index in range(len(test_data)):
        print('\n')

        print('#####DOWNLOAD ', test_data_index, ' BEGINS#####')

        client, endpoint, username, port, key, transfer_type, expected, result, file_name = \
            create_connection_data(test_data, test_data_index)

        sftp_client, sftp_client_transport = connect_to_sftp(endpoint, port, username)
        file_path = create_file_path(sftp_client)

        server_file_size = get_remote_file_size(sftp_client, file_path)

        start = datetime.datetime.now()

        client_file_path = download_single_file(sftp_client, file_path, file_name)

        end = datetime.datetime.now()

        compare_file_size(server_file_size, client_file_path, failures_count)

        sftp_client.close()
        sftp_client_transport.close()

        ddl_time = (end-start).total_seconds()

        print('\n')
        print('FILE PATH: ', file_path)
        print('FILE SIZE IN MB: ', round(server_file_size/(1024*1024), 2))
        print('DOWNLOAD TIME IN SECONDS: ', round(ddl_time, 2))
        print('\n')

        ddl_file_sizes.append(round(server_file_size/(1024*1024), 2))
        ddl_times.append(round(ddl_time, 2))
        ddl_file_paths.append(file_path)

        print('#####DOWNLOAD ', test_data_index, ' ENDS#####')

        test_data_index += 1

    print('\n')

    local_files_list = retrieve_local_files()
    clean_local_files(local_files_list)

    return failures_count, ddl_times, ddl_file_sizes, ddl_file_paths


# main call

test_failures, test_times_list, test_file_sizes, test_file_paths = main_logic()

print('FILE PATHS: ', test_file_paths, '\n')
print('FILE SIZES IN MB', test_file_sizes, '\n')
print('DOWNLOAD TIMES IN SECONDS: ', test_times_list, '\n')

print('\n')

if test_failures != 0:
    print('ALERT: THERE IS ', test_failures, ' FAILING TESTS!')
    sys.exit(test_failures)
else:
    print('NO FAILURES, WE ARE GOOD TO GO!')


