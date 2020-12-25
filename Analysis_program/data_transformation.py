# Importing required libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
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

    # Scaling the numerical variables
    scaler = MinMaxScaler((0, 5))

    scaled_columns = ['Scaled_'+_ for _ in columns]
    scaled_df = pd.DataFrame(scaler.fit_transform(df[columns]), columns=scaled_columns)
    df = df.merge(scaled_df, left_index=True, right_index=True)
    df.drop(columns, axis=1, inplace=True)
    df.rename(columns={'Scaled_'+inputs['impute_column']: inputs['impute_column']}, inplace=True)
    writer.write_report(f'Scaled the features {columns}')
    return df


def label_encode(df, inputs, writer):
    columns = inputs['ordinal_features']

    # Label encode
    ordinal = OrdinalEncoder()
    encoded_columns = ['Encoded_'+_ for _ in columns]
    encoded_df = pd.DataFrame(ordinal.fit_transform(df[columns]), columns=encoded_columns)
    df = df.merge(encoded_df, left_index=True, right_index=True)
    df.drop(columns, axis=1, inplace=True)
    writer.write_report(f'Label encoded the features {columns}')
    return df


def one_hot_encode(df, inputs, writer):
    columns = inputs['nominal_features']

    # One-hot encoding the nominal categorical variables
    dummies_df = pd.get_dummies(df[columns])
    df = df.merge(dummies_df, left_index=True, right_index=True)
    df.drop(columns, axis=1, inplace=True)
    writer.write_report(f'One-hot encoded the features {columns}')
    return df


def data_preprocess(df, inputs, writer):
    # Remove the unwanted features
    if 'features_to_drop' in inputs:
        df = remove_features(df, inputs, writer)

    # Scaling the numerical features
    df = numerical_scaling(df, inputs, writer)

    # Encoding the ordinal categorical features
    df = label_encode(df, inputs, writer)

    # One-hot encoding the nominal categorical features
    df = one_hot_encode(df, inputs, writer)

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
