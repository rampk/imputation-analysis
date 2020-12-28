# Importing the required libraries
from read_write_files import write_performance
import numpy as np
import pandas as pd


def present_evaluation(metrics, inputs, writer):

    if metrics.regression:
        pass
    else:
        metrics_df = pd.DataFrame(columns=['Algorithm', 'Imputation', 'Accuracy_Mean', 'Accuracy_STD',
                                           'AUC_Mean', 'AUC_STD', 'Precision_Mean', 'Precision_STD',
                                           'Recall_Mean', 'Recall_STD'])
        # Mean
        for algorithms in metrics.mean:
            metrics_dict = dict()
            metrics_dict['Algorithm'] = algorithms
            metrics_dict['Imputation'] = 'Mean'
            for metric in metrics.mean[algorithms]:
                metric_mean = metric + '_Mean'
                metric_std = metric + '_STD'
                metrics_dict[metric_mean] = np.mean(metrics.mean[algorithms][metric])
                metrics_dict[metric_std] = np.std(metrics.mean[algorithms][metric])

            # append the info to the DataFrame
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)

        # Median
        for algorithms in metrics.median:
            metrics_dict = dict()
            metrics_dict['Algorithm'] = algorithms
            metrics_dict['Imputation'] = 'Median'
            for metric in metrics.median[algorithms]:
                metric_mean = metric + '_Mean'
                metric_std = metric + '_STD'
                metrics_dict[metric_mean] = np.mean(metrics.median[algorithms][metric])
                metrics_dict[metric_std] = np.std(metrics.median[algorithms][metric])

            # append the info to the DataFrame
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)

        # Mode
        for algorithms in metrics.mode:
            metrics_dict = dict()
            metrics_dict['Algorithm'] = algorithms
            metrics_dict['Imputation'] = 'Mode'
            for metric in metrics.mode[algorithms]:
                metric_mean = metric + '_Mean'
                metric_std = metric + '_STD'
                metrics_dict[metric_mean] = np.mean(metrics.mode[algorithms][metric])
                metrics_dict[metric_std] = np.std(metrics.mode[algorithms][metric])

            # append the info to the DataFrame
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)

    metrics_df.sort_values(by=['Algorithm'], inplace=True)
    write_performance(metrics_df, inputs['run_by'], writer.current_num)
