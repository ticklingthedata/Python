import pandas as pd
import json
import random
import numpy as np
import requests
from requests.auth import HTTPBasicAuth


def create_df_input(test_data_file):

    df_input = pd.read_csv(test_data_file)
    df_input.fillna('', inplace=True)

    return df_input


def retrieve_test_endpoint_params(df_input, df_input_row_index):

    test_name = df_input.iloc[df_input_row_index]['test name']
    test_base_endpoint = df_input.iloc[df_input_row_index]['test base endpoint']
    test_dataset = df_input.iloc[df_input_row_index]['test dataset']
    test_user = df_input.iloc[df_input_row_index]['test user']
    test_password = df_input.iloc[df_input_row_index]['test password']
    test_fields = df_input.iloc[df_input_row_index]['test fields']
    test_start_time = df_input.iloc[df_input_row_index]['test startTime']
    test_end_time = df_input.iloc[df_input_row_index]['test endTime']
    test_predicates = df_input.iloc[df_input_row_index]['test predicates']
    test_file_format = df_input.iloc[df_input_row_index]['test fileFormat']
    test_limit = df_input.iloc[df_input_row_index]['test limit']

    return test_name, test_base_endpoint, test_dataset, test_user, test_password, test_fields, test_start_time, \
        test_end_time, test_predicates, test_file_format, test_limit


def create_test_endpoint(test_base_endpoint, test_dataset, test_fields, test_start_time,
                         test_end_time, test_predicates, test_file_format, test_limit):

    test_endpoint = test_base_endpoint + '/' + test_dataset + '?'

    if test_start_time != '':
        test_endpoint += 'startTime=' + str(round(test_start_time))

    if test_end_time != '':
        test_endpoint += '&endTime=' + str(round(test_end_time))

    if test_fields != '':
        test_endpoint += '&fields=' + str(test_fields)

    if test_predicates != '':
        test_endpoint += '&q=' + str(test_predicates)

    if test_file_format != '':
        test_endpoint += '&fileFormat=' + str(test_file_format)

    if test_limit != '':
        test_endpoint += '&limit=' + str(round(test_limit))

    return test_endpoint


def create_data_access_conditions(conditions_file):

    df_conditions = pd.read_csv(conditions_file)
    data_access_conditions = df_conditions.iloc[0].to_dict()

    return data_access_conditions


def execute_query(endpoint, user, password, file_format):

    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

    try:
        r = requests.get(endpoint, auth=HTTPBasicAuth(user, password), data=None, headers=headers)
        status_code = r.status_code

        if file_format == 'JSON' or file_format == '':

            response = json.loads(r.text)

        else:
            response = r.text

        return status_code, response

    except requests.exceptions.ConnectionError as e:

        print(e.args)


def extract_sample_from_response(response, fields):

    if 'dict' in str(type(response)):
        sample_key_item_counter = 0
        sample = random.choice(response)
        sample_keys = list(sample.keys())

        for sample_keys_item in sample_keys:

            if sample_keys_item in fields:

                sample_key_item_counter += 1

        if sample_key_item_counter != 0:

            if sample_key_item_counter < len(fields.split(',')):

                return False

            else:

                return sample

        else:

            return sample

    else:

        if 'list' in str(type(response)):

            if len(response) > 0:

                sample = random.choice(response)

                return sample

            else:

                return ''


def write_endpoint_status_code_response_sample_to_df_output(df_input):

    df_output = pd.DataFrame(columns=list(df_input.columns))

    for df_input_row_index in range(len(df_input)):

        sample_check = 0

        name, base_endpoint, dataset, user, password, fields, start_time, end_time, predicates, file_format, limit = \
            retrieve_test_endpoint_params(df_input, df_input_row_index)

        endpoint = create_test_endpoint(base_endpoint, dataset, fields, start_time, end_time, predicates, file_format,
                                        limit)
        print('GENERATING TEST DATA FOR TEST#: ', df_input_row_index+1, ' - ', dataset, ' - ', name)

        status_code, response = execute_query(endpoint, user, password, file_format)

        while sample_check == 0:

            sample = extract_sample_from_response(response, fields)

            if sample is not False:

                sample_check = 1

        df_output = df_output.append({'test name': name, 'test base endpoint': base_endpoint, 'test dataset': dataset,
                                      'test user': user, 'test password': password, 'test fields': fields,
                                      'test startTime': start_time, 'test endTime': end_time,
                                      'test predicates': predicates, 'test fileFormat': file_format,
                                      'test limit': limit, 'test endpoint': endpoint, 'test response code': status_code,
                                      'test response sample': sample}, ignore_index=True)

    return df_output


def extract_terms_predicate(predicate):

    predicate_to_list = predicate.split('.')

    response_property = predicate_to_list[0]

    operator = predicate_to_list[1]

    compares_to = predicate_to_list[2]

    return response_property, operator, compares_to


def check_status_code(status_code):

    if status_code == 200:

        return 'PASS'

    else:

        return 'FAIL'


def check_bank_identifier_condition(sample, data_access_conditions, dataset):

    if dataset == 'pooled_modellable_securities_list':

        if 'bank_identifier' in (list(sample.keys())):

            if sample['bank_identifier'] == 'POOLED':

                return 'PASS'

            else:

                return 'FAIL: bank identifier should be Pooled, instead got ' + sample['bank_identifier']

        else:

            return 'FAIL: bank_identifier was not found in the sample keys' + str(list(sample.keys()))

    if dataset == 'modellable_securities_list':

        if 'bank_identifier' in (list(sample.keys())):

            if sample['bank_identifier'] == 'POOLED':

                return 'FAIL: bank identifier should not be Pooled'

            else:

                return 'PASS'

        else:

            return 'FAIL: bank_identifier was not found in the sample keys' + str(list(sample.keys()))

    if dataset == 'pooled_price_observations':

        if ('trade_id_buy' or 'bank_identifier_buy' or 'trade_id_sell' or 'bank_identifier_sell') \
                in (list(sample.keys())):

            return 'FAIL: expected trade_id_buy/bank_identifier_buy or trade_id_sell/bank_identifier_sell ' \
                   'combination'

        else:

            return 'PASS'

    if dataset == 'price_observations':

        if sample != '':

            if len(list(sample.keys())) > 0:

                if (('trade_id_buy' and 'bank_identifier_buy') in (list(sample.keys()))) or \
                        (('trade_id_sell' and 'bank_identifier_sell') in (list(sample.keys()))):

                    return 'PASS'

                else:

                    return 'FAIL: expected trade_id_buy/bank_identifier_buy or trade_id_sell/bank_identifier_sell ' \
                           'combination'

        else:

            return 'PASS'

    if dataset == 'pooled_bond_trades':

        if 'bank_identifier' in (list(sample.keys())):

            return 'FAIL - bank_identifier should not be displayed'

        else:

            return 'PASS'

    else:

        if data_access_conditions['bank_identifier'] == sample['bank_identifier']:

            return 'PASS'

        else:

            return 'FAIL - expected ' + data_access_conditions['bank_identifier'] + \
                   ' found ' + sample['bank_identifier']


def check_fields_condition(sample, fields):

    if sample != '':

        field_list = fields.split(',')
        sample_keys_list = list(sample.keys())
        items_found = list()
        success_counter = 0

        for sample_keys_list_item in sample_keys_list:

            for field_list_item in field_list:

                if sample_keys_list_item == field_list_item:

                    success_counter += 1
                    items_found.append(field_list_item)

        if success_counter == len(sample_keys_list):

            return 'PASS'

        else:

            return 'FAIL - expected' + str(sorted(field_list)) + ' found' + str(sorted(items_found))

    else:

        return 'PASS'


def check_start_time_condition(sample, start_time, dataset):

    if sample != '':

        start_time = round(start_time)

        if dataset == 'bond_quotes':

            date = sample['quote_submission_date']

        if dataset == 'bond_trades':

            date = sample['execution_date']

        if dataset == 'pooled_bond_trades':

            date = sample['execution_date']

        if dataset == 'price_observations' or dataset == 'pooled_price_observations':

            date = sample['observation_date']

        if dataset == 'modellable_securities_list' or dataset == 'pooled_modellable_securities_list':

            date = sample['report_date']

        if start_time <= date:

            return 'PASS'

        else:

            return 'FAIL - ' + str(date) + ' is smaller than ' + str(date)

    else:

        return 'PASS'


def check_end_time_condition(sample, end_time, dataset):

    if sample != '':

        end_time = round(end_time)

        if dataset == 'bond_quotes':

            date = sample['quote_submission_date']

        if dataset == 'bond_trades':

            date = sample['execution_date']

        if dataset == 'pooled_bond_trades':

            date = sample['execution_date']

        if dataset == 'price_observations' or dataset == 'pooled_price_observations':

            date = sample['observation_date']

        if dataset == 'modellable_securities_list' or dataset == 'pooled_modellable_securities_list':

            date = sample['report_date']

        if end_time >= date:

            return 'PASS'

        else:

            return 'FAIL - ' + str(date) + ' is bigger than ' + str(date)

    else:

        return 'PASS'


def check_predicates_condition(sample, predicate):

    response_property, operator, compares_to = extract_terms_predicate(predicate)

    if sample != '':

        if operator == 'eq':
            if str(sample[response_property]) == str(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' and ' + str(compares_to) + ' are not equal'

        if operator == 'gte':
            if int(sample[response_property]) >= int(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is not greater than or equal to ' + str(compares_to)

        if operator == 'lte':
            if int(sample[response_property]) <= int(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is not smaller than or equal to ' + str(compares_to)

        if operator == 'gt':
            if int(sample[response_property]) > int(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is not greater than ' + str(compares_to)

        if operator == 'lt':
            if int(sample[response_property]) < int(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is not smaller than ' + str(compares_to)

        if operator == 'in':
            if str(sample[response_property]) in str(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is not in ' + str(compares_to)

        if operator == 'neq':
            if str(sample[response_property]) != str(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' and ' + str(compares_to) + ' are equal'

        if operator == 'nin':
            if str(sample[response_property]) not in str(compares_to):
                return 'PASS'
            else:
                return 'FAIL - ' + str(sample[response_property]) + ' is in ' + str(compares_to)
    else:

        return 'PASS'


def check_file_format_condition(sample, file_format):

    if sample != '':

        if file_format == 'JSON':

            file_type = 'dict'

        else:

            file_type = 'str'

        if file_type in str(type(sample)):

            return 'PASS'

        else:

            return 'FAIL - expected ' + file_type + ' found ' + str(type(sample))

    else:

        return 'PASS'


def check_limit_condition(response, limit):

    if 'list' in str(type(response)):

        if round(limit) >= len(response):

            return 'PASS'

        else:

            return 'FAIL - expected ' + str(round(limit)) + ' found ' + str(len(response))

    else:

        item_counter = 0
        response_items = response.split(',')

        for item in response_items:

            if ('B1' or 'B2' or 'B3' or 'B4' or 'B5' or 'B6') in item:

                item_counter += 1

        if round(limit) >= item_counter:

            return 'PASS'

        else:

            return 'FAIL - expected ' + str(round(limit)) + ' found ' + str(item_counter)


def test_runner(name, status_code, response, sample, fields, start_time, end_time, predicate,
                                       file_format, limit, data_access_file, dataset):

    fields_condition = ''
    start_time_condition = ''
    end_time_condition = ''
    predicates_condition = ''
    file_format_condition = ''
    data_access_condition = ''

    status_code_condition = check_status_code(status_code)
    limit_condition = check_limit_condition(response, limit)
    data_access = create_data_access_conditions(data_access_file)

    if name == 'ADMIN – dataset and limit' or name == 'BANK USER – dataset and limit':

        if dataset == 'pooled_modellable_securities_list':

            if name == 'ADMIN – dataset and limit':

                data_access_condition = check_bank_identifier_condition(sample, data_access, dataset)

            else:

                if sample['bank_identifier'] == 'POOLED':

                    data_access_condition = 'PASS'

                else:

                    data_access_condition = 'FAIL: user should not have access to' + data_access['bank_identifier']

        if dataset == 'modellable_securities_list':

            if name == 'ADMIN – dataset and limit':

                data_access_condition = check_bank_identifier_condition(sample, data_access, dataset)

            else:

                if sample['bank_identifier'] == data_access['bank_identifier']:

                    data_access_condition = 'PASS'

                else:

                    data_access_condition = 'FAIL: user should not have access to' + data_access['bank_identifier']

        if dataset == 'price_observations' or dataset == 'pooled_price_observations':

            data_access_condition = check_bank_identifier_condition(sample, data_access, dataset)

        else:

            if name == 'BANK USER – dataset and limit':

                data_access_condition = check_bank_identifier_condition(sample, data_access, dataset)

    if name == 'ADMIN – dataset, fields and limit' or name == 'BANK USER – dataset, fields and limit':

        fields_condition = check_fields_condition(sample, fields)

    if name == 'ADMIN – dataset, start time and limit' or name == 'BANK USER – dataset, start time and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)

    if name == 'ADMIN – dataset, end time and limit' or name == 'BANK USER – dataset, end time and limit':

        end_time_condition = check_end_time_condition(sample, end_time, dataset)

    if name == 'ADMIN – dataset, predicates and limit' or name == 'BANK USER – dataset, predicates and limit':

        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, file format and limit' or name == 'BANK USER – dataset, file format and limit':

        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, fields, start time and limit' or \
            name == 'BANK USER – dataset, fields, start time and limit':

        fields_condition = check_fields_condition(sample, fields)
        start_time_condition = check_start_time_condition(sample, start_time, dataset)

    if name == 'ADMIN – dataset, fields, end time and limit' or \
            name == 'BANK USER – dataset, fields, end time and limit':

        fields_condition = check_fields_condition(sample, fields)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)

    if name == 'ADMIN – dataset, fields, predicates and limit' or \
            name == 'BANK USER – dataset, fields, predicates and limit':

        fields_condition = check_fields_condition(sample, fields)
        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, fields, file format and limit' or \
            name == 'BANK USER – dataset, fields, file format and limit':

        file_format_condition = check_file_format_condition(sample, file_format)
        fields_condition = check_fields_condition(sample, fields)

    if name == 'ADMIN – dataset, start time, end time and limit' or \
            name == 'BANK USER – dataset, start time, end time and limit':
        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)

    if name == 'ADMIN – dataset, start time, predicates and limit' or \
            name == 'BANK USER – dataset, start time, predicates and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, start time, file format and limit' or \
            name == 'BANK USER – dataset, start time, file format and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, end time, predicates and limit' or \
            name == 'BANK USER – dataset, end time, predicates and limit':

        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, end time, file format and limit' or \
            name == 'BANK USER – dataset, end time, file format and limit':

        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, predicates, file format and limit' or \
            name == 'BANK USER – dataset, predicates, file format and limit':

        predicates_condition = check_predicates_condition(sample, predicate)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, fields, start time, end time and limit' or \
            name == 'BANK USER – dataset, fields, start time, end time and limit':

        fields_condition = check_fields_condition(sample, fields)
        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)

    if name == 'ADMIN – dataset, fields, start time, predicates and limit' or \
            name == 'BANK USER – dataset, fields, start time, predicates and limit':

        fields_condition = check_fields_condition(sample, fields)
        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, fields, start time, file format and limit' or \
            name == 'BANK USER – dataset, fields, start time, file format and limit':

        fields_condition = check_fields_condition(sample, fields)
        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, start time, end time, predicates and limit' or \
            name == 'BANK USER – dataset, start time, end time, predicates and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)

    if name == 'ADMIN – dataset, start time, end time, file format and limit' or \
            name == 'BANK USER – dataset, start time, end time, file format and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, end time, predicates file format and limit' or \
            name == 'BANK USER – dataset, end time, predicates file format and limit':

        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, start time, end time, predicates, file format and limit' or \
            name == 'BANK USER – dataset, start time, end time, predicates, file format and limit':

        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)
        file_format_condition = check_file_format_condition(sample, file_format)

    if name == 'ADMIN – dataset, fields, start time, end time, predicates, file format and limit' or \
            name == 'BANK USER – dataset, fields, start time, end time, predicates, file format and limit':

        fields_condition = check_fields_condition(sample, fields)
        start_time_condition = check_start_time_condition(sample, start_time, dataset)
        end_time_condition = check_end_time_condition(sample, end_time, dataset)
        predicates_condition = check_predicates_condition(sample, predicate)
        file_format_condition = check_file_format_condition(sample, file_format)

    return status_code_condition, limit_condition, fields_condition, start_time_condition, end_time_condition, \
        predicates_condition, file_format_condition, data_access_condition


def write_check_results_to_df_output(df_input):

    df_output = pd.DataFrame(columns=list(df_input.columns))

    for df_input_row_index in range(len(df_input)):

        name, base_endpoint, dataset, user, password, fields, start_time, end_time, predicates, file_format, limit = \
            retrieve_test_endpoint_params(df_input, df_input_row_index)

        endpoint = df_input.iloc[df_input_row_index]['test endpoint']

        _, response = execute_query(endpoint, user, password, file_format)

        status_code = df_input.iloc[df_input_row_index]['test response code']

        sample = df_input.iloc[df_input_row_index]['test response sample']

        data_access_file = 'bond_quotes_data_access_conditions.csv'

        print('EXECUTING TEST #: ', df_input_row_index+1, ' - ', dataset, ' - ', name)

        status_code_condition, limit_condition, fields_condition, start_time_condition, end_time_condition,\
            predicates_condition, file_format_condition, data_access_condition = \
            test_runner(name, status_code, response, sample, fields, start_time, end_time,
                        predicates, file_format, limit, data_access_file, dataset)

        df_output = df_output.append({'test name': name, 'test base endpoint': base_endpoint, 'test dataset': dataset,
                                      'test user': user, 'test password': password, 'test fields': fields,
                                      'test startTime': start_time, 'test endTime': end_time,
                                      'test predicates': predicates, 'test fileFormat': file_format,
                                      'test limit': limit, 'test endpoint': endpoint,
                                      'test response code': status_code_condition,
                                      'test response sample': sample, 'test fields condition': fields_condition,
                                      'test startTime condition': start_time_condition,
                                      'test endTime condition': end_time_condition,
                                      'test predicates condition': predicates_condition,
                                      'test fileFormat condition': file_format_condition,
                                      'test limit condition': limit_condition,
                                      'test data access condition': data_access_condition}, ignore_index=True)

    return df_output


def write_test_status(df_input):

    df_output = pd.DataFrame(columns=list(df_input.columns))

    for df_input_row_index in range(len(df_input)):

        name, base_endpoint, dataset, user, password, fields, start_time, end_time, predicates, file_format, limit = \
            retrieve_test_endpoint_params(df_input, df_input_row_index)
        endpoint = df_input.iloc[df_input_row_index]['test endpoint']
        sample = df_input.iloc[df_input_row_index]['test response sample']

        response_condition = df_input.iloc[df_input_row_index]['test response code']
        fields_condition = df_input.iloc[df_input_row_index]['test fields condition']
        start_time_condition = df_input.iloc[df_input_row_index]['test startTime condition']
        end_time_condition = df_input.iloc[df_input_row_index]['test endTime condition']
        predicates_condition = df_input.iloc[df_input_row_index]['test predicates condition']
        file_format_condition = df_input.iloc[df_input_row_index]['test fileFormat condition']
        limit_condition = df_input.iloc[df_input_row_index]['test limit condition']
        data_access_condition = df_input.iloc[df_input_row_index]['test data access condition']

        status_list = [response_condition, fields_condition, start_time_condition, end_time_condition,
                       predicates_condition, file_format_condition, limit_condition, data_access_condition]

        for status_list_item in status_list:

            if 'FAIL' in status_list_item:

                test_status_condition = 'FAIL'

            else:

                test_status_condition = 'PASS'

        df_output = df_output.append({'test name': name, 'test base endpoint': base_endpoint, 'test dataset': dataset,
                                      'test user': user, 'test password': password, 'test fields': fields,
                                      'test startTime': start_time, 'test endTime': end_time,
                                      'test predicates': predicates, 'test fileFormat': file_format,
                                      'test limit': limit, 'test endpoint': endpoint,
                                      'test response code': response_condition,
                                      'test response sample': sample, 'test fields condition': fields_condition,
                                      'test startTime condition': start_time_condition,
                                      'test endTime condition': end_time_condition,
                                      'test predicates condition': predicates_condition,
                                      'test fileFormat condition': file_format_condition,
                                      'test limit condition': limit_condition,
                                      'test data access condition': data_access_condition,
                                      'test status': test_status_condition}, ignore_index=True)

    df_output.replace('', np.nan, inplace=True)
    df_output.fillna('N/A', inplace=True)

    return df_output


def main_test(input_file, output_file):

    raw_test_df = create_df_input(input_file)
    generated_test_data_df = write_endpoint_status_code_response_sample_to_df_output(raw_test_df)
    generated_test_results_df = write_check_results_to_df_output(generated_test_data_df)
    generated_test_status_df = write_test_status(generated_test_results_df)
    generated_test_status_df.to_csv(output_file)


inputs = {'bond_quotes': 'bond_quotes_query_api_test_data.csv',
          'bond_trades': 'bond_trades_query_api_test_data.csv',
          'pooled_bond_trades': 'pooled_bond_trades_query_api_test_data.csv',
          'price_observations': 'price_observations_query_api_test_data.csv',
          'pooled_price_observations': 'pooled_price_observations_query_api_test_data.csv',
          'modellable_securities_list': 'modellable_securities_list_query_api_test_data.csv',
          'pooled_modellable_securities_list': 'pooled_modellable_securities_list_query_api_test_data.csv'}

outputs = {'bond_quotes': 'bond_quotes_test_results.csv', 'bond_trades': 'bond_trades_test_results.csv',
           'pooled_bond_trades': 'pooled_bond_trades_test_results.csv',
           'price_observations': 'price_observations_test_results.csv',
           'pooled_price_observations': 'pooled_price_observations_test_results.csv',
           'modellable_securities_list': 'modellable_securities_list_test_results.csv',
           'pooled_modellable_securities_list': 'pooled_modellable_securities_list_test_results.csv'}

# Main

main_test(inputs['bond_quotes'], outputs['bond_quotes'])
main_test(inputs['bond_trades'], outputs['bond_trades'])
main_test(inputs['pooled_bond_trades'], outputs['pooled_bond_trades'])
main_test(inputs['price_observations'], outputs['price_observations'])
main_test(inputs['pooled_price_observations'], outputs['pooled_price_observations'])
main_test(inputs['modellable_securities_list'], outputs['modellable_securities_list'])
main_test(inputs['pooled_modellable_securities_list'], outputs['pooled_modellable_securities_list'])







