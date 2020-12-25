# Importing the libraries
import pandas as pd
from os import listdir
import re


# Write a report
class ReportWriter:
    def __init__(self):
        # Generate a current report name
        def current_file_name():
            last_file = sorted(listdir('../Analysis_steps_performed'))[-1]
            current_report_num = int((re.findall(r'\d+', last_file)[0])) + 1
            return f'../Analysis_steps_performed/Program_run-{current_report_num}.txt'

        self.file_name = current_file_name()
        # Open the file cursor for writing
        self.file_cursor = open(self.file_name, 'w')

    def write_report(self, text):
        self.file_cursor.write(str(text))
        self.file_cursor.write('\n\n')

    def file_close(self):
        self.file_cursor.close()


# Reading the inputs
def read_inputs():
    with open('input.txt', 'r') as file:
        inputs = {i.split('=')[0]: i.split('=')[1].strip() for i in file.readlines()}

    if inputs['features_to_drop'] != 'NA':
        inputs['features_to_drop'] = inputs['features_to_drop'].split(',')
    else:
        del inputs['features_to_drop']

    if inputs['ordinal_features'] != 'NA':
        inputs['ordinal_features'] = inputs['ordinal_features'].split(',')
    else:
        del inputs['ordinal_features']

    if inputs['nominal_features'] != 'NA':
        inputs['nominal_features'] = inputs['nominal_features'].split(',')
    else:
        del inputs['nominal_features']

    if inputs['numerical_cat'] != 'NA':
        inputs['numerical_cat'] = inputs['numerical_cat'].split(',')
    else:
        del inputs['numerical_cat']

    return inputs


# Reading the dataset
def read_dataset(inputs):
    dataset = inputs['dataset']
    delimiter = inputs['delimiter']
    extension = dataset.split('.')[1]
    file_name = '../Data/' + dataset

    if extension == 'csv' or extension == 'data':
        return pd.read_csv(file_name, delimiter=delimiter)
