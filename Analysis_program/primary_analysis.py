# Importing the libraries
from read_write_files import read_inputs, read_dataset, ReportWriter, zip_results
from data_transformation import impute_central_tendency, data_preprocess
from evaluation_modelling import perform_analysis

if __name__ == "__main__":

    # Read inputs
    inputs = read_inputs()

    # Create instance for report writing
    writer = ReportWriter(inputs['run_by'])
    writer.write_report(inputs)

    # Read dataset
    df = read_dataset(inputs)

    # Preprocessing the data
    # Preprocessing before imputation and splitting will reduce the runtime
    df = data_preprocess(df, inputs, writer)

    # Impute with central tendency and create three set of dataframe
    df_mean, df_median, df_mode = impute_central_tendency(df, inputs, writer)

    # Perform the analysis
    perform_analysis(df_mean, df_median, df_mode, inputs, writer)

    # Close the file cursor
    writer.file_close()

    # Wrap up the results
    zip_results(inputs['run_by'], writer.current_num)
