# Importing the libraries
from read_write_files import read_inputs, read_dataset, ReportWriter
from data_transformation import impute_central_tendency, data_preprocess

if __name__ == "__main__":
    # Create instance for report writing
    writer = ReportWriter()

    # Read inputs
    inputs = read_inputs()
    writer.write_report(inputs)

    # Read dataset
    df = read_dataset(inputs)

    # Preprocessing the data
    # Preprocessing before imputation and splitting will reduce the runtime
    df = data_preprocess(df, inputs, writer)
    print(df.head())

    # Impute with central tendency and create three set of dataframe
    df_mean, df_median, df_mode = impute_central_tendency(df, inputs, writer)

    #print(df_mean.Age[5])
    #print(df_median.Age[5])
    #print(df_mode.Age[5])
    #print(df_mean.columns)

    writer.file_close()
