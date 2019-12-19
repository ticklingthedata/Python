import boto3
import pandas as pd
import flatten_dict
from tabulate import tabulate

print('\n')
profile = input('PROFILE: ') #'ice-dev-developer'


def connect_to_s3(s3_profile):

    """
    Connects to s3 and gets the session used in search_s3_files function.
    """
    s3_session = boto3.Session(profile_name=s3_profile)
    if s3_session.profile_name != '':
        print('\n')
        print('SUCCESSFULLY CONNECTED TO S3 AS', s3_session.profile_name, '.')
        print('\n')

    return s3_session


def create_ec2_client_from_s3_session(s3_session):
    """
    Creates s3 client from s3 session.
    """
    # Create the resource

    s3_ec2_client = s3_session.client('ec2')

    return s3_ec2_client


def create_health_client_from_s3_session(s3_session):
    """
    Creates s3 client from s3 session.
    """
    # Create the resource

    s3_health_client = s3_session.client('health')

    return s3_health_client


def describe_all_instances(s3_ec2_client):

    all_described_ec2_instances = s3_ec2_client.describe_instances()

    return all_described_ec2_instances


def describe_all_affected_entities(s3_health_client):

    print(s3_health_client.describe_affected_entities())


def flatten_raw_instance_list(all_described_ec2_instances):

    flattened_ec2_instances = flatten_dict.flatten(all_described_ec2_instances)

    return flattened_ec2_instances


def create_instance_df(all_described_ec2_instances):

    df_ec2_instances = pd.DataFrame.from_dict(data=all_described_ec2_instances, orient='columns')

    return df_ec2_instances


def transform_instance_df(df_ec2_instances):

    transformed_df_ec2_instances = df_ec2_instances['Reservations']
    transformed_df_ec2_instances.columns = ['instance_info']

    return transformed_df_ec2_instances


def create_ec2_instance_list_info(transformed_df_ec2_instances):

    ec2_instance_list_info = list()

    for row_index in range(len(transformed_df_ec2_instances)):
        ec2_instance = flatten_raw_instance_list(transformed_df_ec2_instances.iloc[row_index]['instance_info'])
        for key in ec2_instance.keys():
            if 'Instance' in key[0]:
                ec2_instance_list_info.append(ec2_instance[key])

    return ec2_instance_list_info


def create_ec2_instance_id_list(ec2_instance_list_info):

    ec2_instance_id_list = list()

    for index in range(len(ec2_instance_list_info)):
        ec2_instance_id_list.append(ec2_instance_list_info[index][0]['InstanceId'])

    return ec2_instance_id_list


def create_df_ec2_statuses(ec2_instance_id_list):

    df_ec2_statuses = pd.DataFrame(columns=['InstanceId', 'InstanceState', 'InstanceStatus', 'SystemStatus'])

    for first_level_key in ec2_instance_id_list:
        if first_level_key == 'InstanceStatuses':
            for second_level_key in range(len(ec2_instance_id_list[first_level_key])):
                df_ec2_statuses = df_ec2_statuses.append({'InstanceId': ec2_instance_id_list[first_level_key]
                                                          [second_level_key]['InstanceId'],
                                                          'InstanceState': ec2_instance_id_list[first_level_key][second_level_key]
                                                          ['InstanceState'],
                                                          'InstanceStatus': ec2_instance_id_list[first_level_key]
                                                          [second_level_key]['InstanceStatus'],
                                                          'SystemStatus': ec2_instance_id_list[first_level_key][second_level_key]
                                                          ['SystemStatus']},
                                                         ignore_index=True)

    return df_ec2_statuses


def main():

    session = connect_to_s3(profile)
    client = create_ec2_client_from_s3_session(session)
    raw_instance_list = describe_all_instances(client)
    flattened_instance_list = flatten_raw_instance_list(raw_instance_list)
    df_instances = create_instance_df(flattened_instance_list)
    transformed_df_instances = transform_instance_df(df_instances)
    instance_list_info = create_ec2_instance_list_info(transformed_df_instances)
    instance_id_list = create_ec2_instance_id_list(instance_list_info)
    statuses = client.describe_instance_status(InstanceIds=instance_id_list)
    ec2_statuses = create_df_ec2_statuses(statuses)
    ec2_statuses.to_csv('ec2_instance_status.csv')
    print(tabulate(ec2_statuses, headers='keys', tablefmt='fancy_grid'))


main()

