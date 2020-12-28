# Importing the required libraries
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def evaluate_performance(train_data, test_data, train_label, test_label, inputs):
    print(inputs['regression'])


def perform_analysis(df_mean, df_median, df_mode, inputs, writer):
    # Splitting the predictor and target variable
    target_variable = inputs['target_variable']
    imputed_variable = inputs['impute_column']
    x_mean = df_mean.drop([target_variable], axis=1)
    x_median = df_median.drop([target_variable], axis=1)
    x_mode = df_mode.drop([target_variable], axis=1)
    y = df_mean[target_variable]

    # Splitting the imputed column into bins
    # Due to scaling values will always range between 0 to 5
    bins = [-0.9, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_mean['bin'] = pd.cut(df_mean[imputed_variable], bins=bins, labels=labels)

    # Using stratified to perform cross validation
    shuffle_split = StratifiedShuffleSplit(n_splits=10, random_state=42, test_size=0.2)

    for train_index, test_index in shuffle_split.split(df_mean, df_mean['bin']):
        train_mean = x_mean.iloc[train_index]
        train_median = x_median.iloc[train_index]
        train_mode = x_mode.iloc[train_index]
        train_y = y.iloc[train_index]

        test_mean = x_mean.iloc[test_index]
        test_median = x_median.iloc[test_index]
        test_mode = x_mode.iloc[test_index]
        test_y = y.iloc[test_index]

        evaluate_performance(train_mean, test_mean, train_y, test_y, inputs)





