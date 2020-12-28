# Importing the required libraries
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
# Machine Learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Metrics calculations
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class Metrics:
    def __init__(self, regression):
        if regression == 'True':
            pass
        else:
            self.mean = {'RandomForest': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                         'LogisticRegression': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                         'KNeighbors': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                         'StateVector': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []}}
            self.median = {'RandomForest': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                           'LogisticRegression': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                           'KNeighbors': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                           'StateVector': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []}}
            self. mode = {'RandomForest': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                          'LogisticRegression': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                          'KNeighbors': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []},
                          'StateVector': {'Accuracy': [], 'AUC': [], 'Precision': [], 'Recall': [], 'F1': []}}

    def add_results(self, results, impute_type):
        if impute_type == 'mean':
            for algorithms in results:
                for metrics in results[algorithms]:
                    self.mean[algorithms][metrics].append((results[algorithms][metrics]))
        elif impute_type == 'median':
            for algorithms in results:
                for metrics in results[algorithms]:
                    self.median[algorithms][metrics].append((results[algorithms][metrics]))
        else:
            for algorithms in results:
                for metrics in results[algorithms]:
                    self.mode[algorithms][metrics].append((results[algorithms][metrics]))


def test_performance(model, test_data, test_label):
    results = dict()
    predicted_label = model.predict(test_data)
    results['Accuracy'] = accuracy_score(test_label, predicted_label)
    results['AUC'] = roc_auc_score(test_label, predicted_label)
    results['Precision'] = precision_score(test_label, predicted_label)
    results['Recall'] = recall_score(test_label, predicted_label)
    results['F1'] = f1_score(test_label, predicted_label)
    return results


def evaluate_performance(train_data, test_data, train_label, test_label, inputs):
    results = dict()
    if inputs['regression'] == 'True':
        pass
    else:
        # Train RandomForest algorithm
        rf = RandomForestClassifier()
        rf.fit(train_data, train_label)
        results['RandomForest'] = test_performance(rf, test_data, test_label)

        # Train LogisticRegression algorithm
        lr = LogisticRegression()
        lr.fit(train_data, train_label)
        results['LogisticRegression'] = test_performance(lr, test_data, test_label)

        # Train KNeighborsClassifier algorithm
        knn = KNeighborsClassifier()
        knn.fit(train_data, train_label)
        results['KNeighbors'] = test_performance(knn, test_data, test_label)

        # Train StateVectorMachine algorithm
        svm = SVC()
        svm.fit(train_data, train_label)
        results['StateVector'] = test_performance(svm, test_data, test_label)

        return results


def perform_analysis(df_mean, df_median, df_mode, inputs, writer):
    # Splitting the predictor and target variable
    target_variable = inputs['target_variable']
    imputed_variable = inputs['impute_column']
    x_mean = df_mean.drop([target_variable], axis=1)
    x_median = df_median.drop([target_variable], axis=1)
    x_mode = df_mode.drop([target_variable], axis=1)
    y = df_mean[target_variable]

    # Creating an instance of metrics class to store evaluation results
    metrics = Metrics(inputs['regression'])

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

        mean_temp = evaluate_performance(train_mean, test_mean, train_y, test_y, inputs)
        metrics.add_results(mean_temp, 'mean')

        median_temp = evaluate_performance(train_median, test_median, train_y, test_y, inputs)
        metrics.add_results(median_temp, 'median')

        mode_temp = evaluate_performance(train_mode, test_mode, train_y, test_y, inputs)
        metrics.add_results(mode_temp, 'mode')

    print(metrics.mean)
    print(metrics.median)
    print(metrics.mode)
