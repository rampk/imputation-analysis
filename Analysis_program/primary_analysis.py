# Importing the libraries
from read_write_files import read_inputs, read_dataset, ReportWriter
from data_transformation import impute_central_tendency

if __name__ == "__main__":
    # Create instance for report writing
    writer = ReportWriter()

    # Read inputs
    inputs = read_inputs()
    writer.write_report(inputs)

    # Read dataset
    df = read_dataset(inputs['dataset'], inputs['delimiter'])

    # Impute with central tendency and create three set of dataframe
    df_mean, df_median, df_mode = impute_central_tendency(df, inputs['impute_column'], writer)

    writer.file_close()
