"""
 This modules exists to provide the necessary functions to handle the
 testing environment side-operations like loading environments variables,
 cleaning data, etc.
 It also decouples these functions from other clients to make sure the
 prior only handle client specific tasks/functions. This will make them
 reusable by any client if needed.
"""

import pandas as pd


def create_test_config_df(table):

    # Converts the spec table (Markdown) to a dataframe (Pandas)

    test_config_df = pd.DataFrame(table, columns=table.headers)

    return test_config_df


def retrieve_test_config_properties(test_config_df, row_index):

    # Extracts cell values from dataframe

    instance = test_config_df.iloc[row_index]['instance']
    customer_id = test_config_df.iloc[row_index]['customerId']
    institution = test_config_df.iloc[row_index]['institution']
    username = test_config_df.iloc[row_index]['username']
    password = test_config_df.iloc[row_index]['password']
    expected_status_code = int(test_config_df.iloc[row_index]['expectedStatusCode'])

    return instance, customer_id, institution, username, password, expected_status_code


def retrieve_request_id(response):

    # Do not try-except key error. Test should explicitly fail if key is not found

    request_id = response['RequestId']

    return request_id


def retrieve_holder_name(response):

    # Do not try-except key error. Test should explicitly fail if key is not found

    holder_name = response['Accounts'][0]['Holder']['Name']

    return holder_name


def retrieve_holder_email(response):

    # Do not try-except key error. Test should explicitly fail if key is not found

    holder_email = response['Accounts'][0]['Holder']['Email']

    return holder_email


def retrieve_account_number_list(response):

    account_number_list = list()

    # Do not try-except key error. Test should explicitly fail if key is not found

    for account in response['Accounts']:

        if account['Category'] == "Operations":

            account_number = {"AccountNumber": account["AccountNumber"]}
            account_number_list.append(account_number)

    return account_number_list


def retrieve_balance_list(response):

    balance_list = list()

    # Do not try-except key error. Test should explicitly fail if key is not found

    for account in response['Accounts']:

        if account['Currency'] == "USD":
            balance = {"Balance": account["Balance"]["Current"]}
            balance_list.append(balance)

    return balance_list


def retrieve_biggest_credit_trx_id(response):

    df_credits = pd.DataFrame(columns=['Id', 'Credit'])

    # Do not try-except key error. Test should explicitly fail if key is not found

    for account in response['Accounts']:

        if account['Category'] == "Credits":

            for transaction in account['Transactions']:

                if transaction['Credit'] is not None:

                    credit_transaction = {'Id': transaction['Id'], 'Credit': float(transaction['Credit'])}
                    df_credits = df_credits.append(credit_transaction, ignore_index=True)

    # Relatively big number of comparisons here. Pandas makes it easier.
    # Store all values and Ids when it's a credit transaction. Call idxmax()
    # to retrieve the biggest value row index. Get the Id using the prior.

    biggest_credit_trx_id_row_index = df_credits['Credit'].astype('float64').idxmax()
    biggest_credit_trx_id = df_credits.iloc[biggest_credit_trx_id_row_index]['Id']

    return biggest_credit_trx_id


def set_desired_format(login_id, request_id, holder_name, holder_email, account_number_list, balance_list,
                       biggest_credit_trx_id):

    desired_format = {
        "LoginId": login_id,
        "RequestId": request_id,
        "Holder": {
            "Name": holder_name,
            "Email": holder_email
        },
        "OperationAccounts": account_number_list,
        "USDAccounts": balance_list,
        "BiggestCreditTrxId": biggest_credit_trx_id
    }

    return desired_format
