import pexpect
import sys
import pandas as pd
import datetime
import pysnooper


def download_files():

    df_test = pd.read_csv('test_data.csv')

    durations_list = list()

    for test_index in range(len(df_test)):

        print(test_index, len(df_test))

        password = df_test.iloc[test_index]['password']
        connection_command = df_test.iloc[test_index]['connection command']
        local_path = df_test.iloc[test_index]['local path']
        download_command = df_test.iloc[test_index]['download command'] + ' ' + local_path

        print('\n')
        print('STARTING DOWNLOAD FILE ')
        start = datetime.datetime.now()

        child = pexpect.spawn(command=connection_command)
        child.logfile = sys.stdout.buffer
        child.expect('Password:', timeout=15)
        child.sendline(password)
        child.expect('sftp> ', timeout=15)
        child.sendline(download_command)
        child.expect('sftp> ', timeout=15)
        child.sendline('quit')
        end = datetime.datetime.now()

        test_duration = (end-start).total_seconds()

        durations_list.append(round(test_duration/60, 2))

    print(durations_list)


if __name__ == '__main__':
    download_files()



