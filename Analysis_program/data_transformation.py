import pandas as pd


# Impute and return three dataframes
def impute_central_tendency(df, column, writer):
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
