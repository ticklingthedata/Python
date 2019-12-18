"""
This is the automation runner. It contains all the automation steps executed
in the order they appear.
"""

# Import cascade yaay!
# Helps visually identifying what has been imported and from where.

from getgauge.python \
    import step, \
    continue_on_failure

from steps.utils.utils \
    import create_test_config_df, \
    retrieve_test_config_properties,\
    set_desired_format, \
    retrieve_request_id, \
    retrieve_holder_name, \
    retrieve_holder_email, \
    retrieve_account_number_list, \
    retrieve_balance_list, \
    retrieve_biggest_credit_trx_id

from steps.clients.accounts \
    import get_request_id, \
    get_request_id_endpoint_builder, \
    get_request_id_payload_builder, \
    authorize_request_payload_builder, \
    authorize_request_id, \
    get_accounts_detail_payload_builder, \
    get_account_details


# endpoint will be used in get_request_id and authorize_request_id.
# Avoid state management using return values between functions to decouple test steps.
# Global variables are not evil. Not giving them credit is the real EVIL (666)!

endpoint = ""
requests_to_authorize = list()
authorized_requests = list()
completed_operations = list()


@continue_on_failure
@step("For each test case, get RequestId using the following connection information <table>")
def get_request_ids_to_authorize(table):

    global requests_to_authorize
    global endpoint

    test_config_df = create_test_config_df(table=table)

    # Each row in the table could be seen as a test with its own request data.
    # This allows to execute multiple tests with different inputs in the same steps if needed

    for row_index in range(len(test_config_df)):

        instance, customer_id, institution, username, password, expected_status_code = \
            retrieve_test_config_properties(test_config_df=test_config_df, row_index=row_index)

        endpoint = get_request_id_endpoint_builder(instance=instance, customer_id=customer_id)
        payload = get_request_id_payload_builder(institution=institution, username=username, password=password)

        print('\nTesting with user: %s' % username)

        status_code, response = get_request_id(endpoint=endpoint, payload=payload)

        # This will gather the request IDs to be authorized in the next step.
        # Creates a list of key-value pairs from request IDs and challenge questions

        if status_code != 401:

            request_challenge = response["SecurityChallenges"][0]["Prompt"]
            request_id = response['RequestId']

            request_id_challenge_pair = {'requestId': request_id, 'challenge': request_challenge}
            requests_to_authorize.append(request_id_challenge_pair)

        assert (status_code == expected_status_code), \
            "Expected status code is %s ### Request returned status code %s" % (expected_status_code, status_code)


@continue_on_failure
@step("Authorize each RequestId (!=401) using the following challenges: <table>")
def authorize_request_ids(table):

    global requests_to_authorize
    global authorized_requests
    global endpoint

    challenges = create_test_config_df(table).to_dict(orient='records')

    for requests_to_authorize_index in range(len(requests_to_authorize)):

        # For each request, this will find the appropriate challenge to answer.
        # Then, every request_id will be sent to authorization process.

        for challenges_index in range(len(challenges)):

            answer_list = list()

            if requests_to_authorize[requests_to_authorize_index]['challenge'] == \
                    challenges[challenges_index]['Question']:

                request_id = requests_to_authorize[requests_to_authorize_index]['requestId']

                print('\nAuthorizing: %s' % request_id)

                question = challenges[challenges_index]['Question']
                answer_list.append(challenges[challenges_index]['Answer'])

                security_responses = {question: answer_list}

                authorize_request_payload = \
                    authorize_request_payload_builder(request_id=request_id, security_responses=security_responses)

                authorize_request_id_status_code = \
                    authorize_request_id(endpoint=endpoint, payload=authorize_request_payload)

                assert (authorize_request_id_status_code == 200), \
                    "Not authorized: %d " % authorize_request_id_status_code

                authorized_requests.append(request_id)

                break


@continue_on_failure
@step("Get the accounts detail for each authorized request")
def get_accounts_detail():

    global authorized_requests
    global completed_operations
    global endpoint

    # For each authorized request_id, details will be gathered to be filtered further.

    for request_id in authorized_requests:

        accounts_detail_endpoint = endpoint.replace('Authorize', 'GetAccountsDetail')
        accounts_detail_payload = get_accounts_detail_payload_builder(request_id)

        print('\nGetting account detail for requestId: %s' % request_id)

        accounts_detail_status_code, accounts_detail_response = \
            get_account_details(accounts_detail_endpoint, accounts_detail_payload)

        if accounts_detail_status_code == 200:

            completed_operations.append(accounts_detail_response)

        # Operation in progress is not a failure.
        # But, only 200 responses will make it to the next step

        assert (accounts_detail_status_code == 200 or accounts_detail_status_code == 202), \
            "Request failed: %s" % accounts_detail_status_code


@continue_on_failure
@step("Filter the details and present the results in the desired format")
def filter_account_detail():

    global completed_operations

    # I have no loginId. Working with requestId only. Not sure where to get it...
    # Build the final format and displays it for each completed operation

    for completed_operation in completed_operations:

        login_id = None
        request_id = retrieve_request_id(response=completed_operation)
        holder_name = retrieve_holder_name(response=completed_operation)
        holder_email = retrieve_holder_email(response=completed_operation)
        account_number_list = retrieve_account_number_list(response=completed_operation)
        balance_list = retrieve_balance_list(response=completed_operation)
        biggest_credit_trx_id = retrieve_biggest_credit_trx_id(response=completed_operation)

        desired_format = set_desired_format(login_id=login_id, request_id=request_id, holder_name=holder_name,
                                            holder_email=holder_email, account_number_list=account_number_list,
                                            balance_list=balance_list, biggest_credit_trx_id=biggest_credit_trx_id)

        print('Result in the desired format: %s' % desired_format)
