"""
 This modules exists to provide the necessary functions to handle the
 api requests related to the accounts component.
"""

import json
import requests

headers = {'Content-type': 'application/json'}


def get_request_id_endpoint_builder(instance, customer_id):

    request_endpoint = 'https://' + instance + '.private.fin.ag/v3/' + customer_id + '/BankingServices/Authorize'

    return request_endpoint


def get_request_id_payload_builder(institution, username, password):

    request_payload = json.dumps({
        "Institution": institution,
        "Username": username,
        "Password": password

    })

    return request_payload


def get_request_id(endpoint, payload):

    try:

        r = requests.post(url=endpoint, data=payload, headers=headers)
        status_code = r.status_code
        response = json.loads(r.text)

        return status_code, response

    except requests.exceptions.ConnectionError as e:

        print(e.args)


def authorize_request_payload_builder(request_id, security_responses):

    authorize_request_payload = json.dumps({
        "RequestId": request_id,
        "SecurityResponses": security_responses
        })

    return authorize_request_payload


def authorize_request_id(endpoint, payload):

    try:

        r = requests.post(url=endpoint, data=payload, headers=headers)
        status_code = r.status_code

        return status_code

    except requests.exceptions.ConnectionError as e:

        print(e.args)


def get_accounts_detail_payload_builder(request_id):

    accounts_detail_payload = json.dumps({
        "RequestId": request_id,
        "WithBalance": "true",
        "WithAccountIdentity": "true",
        "WithTransactions": "true",
        "DaysOfTransactions": 90
        })

    return accounts_detail_payload


def get_account_details(endpoint, payload):

    try:

        r = requests.post(url=endpoint, data=payload, headers=headers)
        status_code = r.status_code
        response = json.loads(r.text)

        return status_code, response

    except requests.exceptions.ConnectionError as e:

        print(e.args)
