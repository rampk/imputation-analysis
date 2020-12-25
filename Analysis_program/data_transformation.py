# Importing required libraries
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def remove_features(df, inputs, writer):
    # List of columns
    columns = df.columns.to_list()
    columns = [_ for _ in columns if _ not in inputs['features_to_drop']]
    writer.write_report(f'Removed the features {inputs["features_to_drop"]}')
    return df[columns]


def numerical_scaling(df, inputs, writer):
    # Extract columns to scale
    columns = df.select_dtypes(exclude='object').columns.tolist()
    if 'numerical_cat' in inputs:
        columns = [_ for _ in columns if _ not in inputs['numerical_cat']]

    columns.remove(inputs['target_variable'])
    columns.remove(inputs['impute_column'])

    # Scaling the numerical variables
    scaler = MinMaxScaler((0, 5))

    scaled_columns = ['Scaled_'+_ for _ in columns]
    scaled_df = pd.DataFrame(scaler.fit_transform(df[columns]), columns=scaled_columns)
    df = df.merge(scaled_df, left_index=True, right_index=True)
    df.drop(columns, axis=1, inplace=True)
    writer.write_report(f'Scaled the features {columns}')
    return df


def data_preprocess(df, inputs, writer):

    # Remove the unwanted features
    if 'features_to_drop' in inputs:
        df = remove_features(df, inputs, writer)

    # Scaling the numerical features
    df = numerical_scaling(df, inputs, writer)

    return df


def impute_central_tendency(df, inputs, writer):
    column = inputs['impute_column']

    # Number of Null records
    total_records = df.shape[0]
    null_records = df[column].isnull().sum()
    percentage = null_records/total_records
    record = f'Total records = {total_records}\nNull records = {null_records}\nPercentage of null = {percentage}\n'
    writer.write_report(record)

    # Create, impute, and return three dataframe
    df_mean = df.copy()
    df_mode = df.copy()
    df_median = df.copy()

    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]

    df_mean[column] = df_mean[column].fillna(mean)
    df_median[column] = df_median[column].fillna(median)
    df_mode[column] = df_mode[column].fillna(mode)
    record = f'mean = {mean}\nmedian = {median}\nmode = {mode}\n'
    writer.write_report(record)

    return df_mean, df_median, df_mode
