import boto3
import json
import pandas as pd
from tabulate import tabulate
import time
import requests
import arrow


def connect_to_s3(s3_profile):

    s3_session = boto3.Session(profile_name=s3_profile)
    if s3_session.profile_name != '':
        print('\n')
        print('SUCCESSFULLY CONNECTED TO S3 AS', s3_session.profile_name, '.')
        print('\n')

    return s3_session


def create_sqs_client_from_s3_session(s3_session):

    s3_sqs_client = s3_session.client('sqs')

    return s3_sqs_client


def create_sns_client_from_s3_session(s3_session):

    s3_sns_client = s3_session.client('sns')

    return s3_sns_client


def get_sns_topics(s3_sns_client):
    sns_topics = s3_sns_client.list_topics()

    return sns_topics


def get_sqs_queues(s3_sqs_client):
    sqs_queues = s3_sqs_client.list_queues()

    return sqs_queues


def create_sns_topic_list(sns_topics):

    topics = list()

    for topic_list_index in sns_topics['Topics']:
        topics.append(topic_list_index['TopicArn'])

    return topics


def create_sqs_message_list(s3_sqs_client, sqs_queues):

    messages = list()

    for queue_list_index in sqs_queues['QueueUrls']:
        queue_messages = s3_sqs_client.receive_message(QueueUrl=queue_list_index, MaxNumberOfMessages=10)
        if 'Messages' in queue_messages.keys():
            for queue_index in queue_messages['Messages']:
                for message_index in queue_index.keys():
                    if message_index == 'Body':
                        messages.append(json.loads(queue_index[message_index]))

    return messages


def send_slack_notification(webhook_url, message):

    attachments = [
        {
            "color": "#ff0000",
            "title": "Airflow Alert",
            "text": "See message below",
            "fields": [
                {
                    "title": "Priority",
                    "value": "High",
                    "short": False
                }
            ],
            "thumb_url": "https://d33np9n32j53g7.cloudfront.net/assets/stacks/airflow/img/airflow-stack-220x234-613461a0bb1df0b065a5b69146fbe061.png",
        }
    ]

    if 'ALERT' in message:
        message = '*`' + message + '`*'
        response = requests.post(webhook_url,
                                 json={'text': message, 'attachments': attachments},
                                 headers={'Content-Type': 'application/json'})
    else:
        message = '```' + message + '```'
        response = requests.post(webhook_url,
                                 json={'text': message},
                                 headers={'Content-Type': 'application/json'})

    if response.status_code != 200:
        raise ValueError('Request to slack returned an error %s, the response is:\n%s' %
                         (response.status_code, response.text))


def write_df_tpcs_msgs(webhook_url,topics, messages):

    df_messages = pd.DataFrame(columns=['TopicArn', 'Message', 'Received DateTime', 'Needs Action'])

    for topic in topics:
        for message_index in range(len(messages)):
            if topic == messages[message_index]['TopicArn']:

                raw_message = json.loads(messages[message_index]['Message'])
                message = json.dumps(raw_message, indent=4, sort_keys=True)
                received_datetime = arrow.now().format('YYYY-MM-DD HH:mm:ss ZZ')

                send_slack_notification(webhook_url,
                                        '\U0001F525 \U0001F525 \U0001F525 NEW ALERT ' + received_datetime + ' ' +
                                        '\U0001F525 \U0001F525 \U0001F525')
                send_slack_notification(webhook_url, message)
                if 'alert' in topic:
                    needs_action = 'Yes'
                else:
                    needs_action = 'No'
                df_messages = df_messages.append({'TopicArn': topic, 'Message': message, 'Received DateTime':
                                                 received_datetime, 'Needs Action': needs_action}, ignore_index=True)

    return df_messages


def print_df_tpcs_msgs(df_messages):

    print(tabulate(df_messages, headers='keys', tablefmt='fancy_grid'))


def main():
    # Generate your webhook from Slack website
    webhook = 'https://hooks.slack.com/services/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'
    execution_frequency = 300

    while True:

        session = connect_to_s3('ice-dev-developer')
        sns_client = create_sns_client_from_s3_session(session)
        sqs_client = create_sqs_client_from_s3_session(session)

        raw_topics = get_sns_topics(sns_client)
        raw_queues = get_sqs_queues(sqs_client)

        topic_list = create_sns_topic_list(raw_topics)
        message_list = create_sqs_message_list(sqs_client, raw_queues)

        df_topics_messages = write_df_tpcs_msgs(webhook, topic_list, message_list)

        print_df_tpcs_msgs(df_topics_messages)

        print('\n', 'NEXT EXECUTION IN ', execution_frequency,  ' SECONDS...')

        time.sleep(execution_frequency)


main()

