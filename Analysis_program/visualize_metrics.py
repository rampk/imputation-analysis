import os
import matplotlib.pyplot as plt


def plot_performance(values, title, folder):
    file_name = f'{folder}/{title}.png'
    xticks = ['Mean', 'Median', 'Mode']
    plt.boxplot(values)
    plt.violinplot(values)
    plt.xticks(range(1, 4), xticks)
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()


def visualize_performance(metrics, inputs, writer):
    # Create a directory for storing images
    run_by = inputs['run_by']
    dir_name = f'../Results/Images/{run_by}_images_{writer.current_num}'
    os.makedirs(dir_name)

    # Visualize the metrics
    impute_types = [metrics.mean, metrics.median, metrics.mode]

    # Visualize Accuracy
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Accuracy'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Accuracy'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Accuracy'])
            else:
                svm.append(impute[algorithm]['Accuracy'])

    plot_performance(forest, "RandomForest_Accuracy", dir_name)
    plot_performance(logistic, "LogisticRegression_Accuracy", dir_name)
    plot_performance(knn, "KNN_Accuracy", dir_name)
    plot_performance(svm, "SVM_Accuracy", dir_name)

    # Visualize AUC
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['AUC'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['AUC'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['AUC'])
            else:
                svm.append(impute[algorithm]['AUC'])

    plot_performance(forest, "RandomForest_AUC", dir_name)
    plot_performance(logistic, "LogisticRegression_AUC", dir_name)
    plot_performance(knn, "KNN_AUC", dir_name)
    plot_performance(svm, "SVM_AUC", dir_name)

    # Visualize Precision
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Precision'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Precision'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Precision'])
            else:
                svm.append(impute[algorithm]['Precision'])

    plot_performance(forest, "RandomForest_Precision", dir_name)
    plot_performance(logistic, "LogisticRegression_Precision", dir_name)
    plot_performance(knn, "KNN_Precision", dir_name)
    plot_performance(svm, "SVM_Precision", dir_name)

    # Visualize Recall
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['Recall'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['Recall'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['Recall'])
            else:
                svm.append(impute[algorithm]['Recall'])

    plot_performance(forest, "RandomForest_Recall", dir_name)
    plot_performance(logistic, "LogisticRegression_Recall", dir_name)
    plot_performance(knn, "KNN_Recall", dir_name)
    plot_performance(svm, "SVM_Recall", dir_name)

    # Visualize F1
    forest = []
    logistic = []
    knn = []
    svm = []

    for impute in impute_types:
        for algorithm in impute:
            if algorithm == 'RandomForest':
                forest.append(impute[algorithm]['F1'])
            elif algorithm == 'LogisticRegression':
                logistic.append(impute[algorithm]['F1'])
            elif algorithm == 'KNeighbors':
                knn.append(impute[algorithm]['F1'])
            else:
                svm.append(impute[algorithm]['F1'])

    plot_performance(forest, "RandomForest_F1", dir_name)
    plot_performance(logistic, "LogisticRegression_F1", dir_name)
    plot_performance(knn, "KNN_F1", dir_name)
    plot_performance(svm, "SVM_F1", dir_name)
