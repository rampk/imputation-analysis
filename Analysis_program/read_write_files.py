# Importing the libraries
from zipfile import ZipFile
import pandas as pd
import re
import os


# Write a report
class ReportWriter:
    def __init__(self, run_by):
        # Generate a current report name
        def current_file_name():
            files = os.listdir('../Analysis_steps_performed')
            if not files:
                return 1, f'../Analysis_steps_performed/{run_by}_program_run_1.txt'
            else:
                last_file = sorted(files)[-1]
                current_report_num = int((re.findall(r'\d+', last_file)[0])) + 1
                return current_report_num, f'../Analysis_steps_performed/{run_by}_program_run_{current_report_num}.txt'

        self.current_num, self.file_name = current_file_name()
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


def document_performance(metrics, run_by, current_num):
    file_name = f'../Results/Performance/{run_by}_performance_results_{current_num}.txt'
    with open(file_name, 'w') as file:
        file.writelines(f'Mean={metrics.mean}\n')
        file.writelines(f'Median={metrics.median}\n')
        file.writelines(f'Mode={metrics.mode}')


def write_performance(df, run_by, current_num):
    file_name = f'../Results/Performance/{run_by}_performance_results_{current_num}.csv'
    df.to_csv(file_name, index=False)


def zip_results(run_by, current_num):
    file_name = f'../Results/{run_by}_analysis_results_{current_num}.zip'
    performance_csv = f'../Results/Performance/{run_by}_performance_results_{current_num}.csv'
    performance_txt = f'../Results/Performance/{run_by}_performance_results_{current_num}.txt'
    steps = f'../Analysis_steps_performed/{run_by}_program_run_{current_num}.txt'
    input_file = f'{run_by}_input_{current_num}.txt'

    with ZipFile(file_name, 'w') as zip_file:
        zip_file.write(performance_csv, os.path.basename(performance_csv))
        zip_file.write(performance_txt, os.path.basename(performance_txt))
        zip_file.write(steps, os.path.basename(steps))
        zip_file.write('input.txt', input_file)
