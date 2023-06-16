from data_preprocessor import DataPreprocessor
from data_builder import DataBuilder
from data_etl import DataETL
from model.XGBoost import XGBoost

import pandas as pd

def load_data():
    etl = DataETL()
    etl.use_local_data("research/data_given/")
    etl.join_tables()
    return etl.get_df()

def preprocess(df:pd.DataFrame):
    dp = DataPreprocessor(df)
    return dp.get_df()

def train(df:pd.DataFrame):
    """
    Takes in the joined and preprocess full df.
    """
    databuilder = DataBuilder(df)
    X_train = databuilder.get_transformed_X_train()
    y_train = databuilder.get_y_train()
    model = XGBoost()
    print(y_train.value_counts())
    model.train(X_train, y_train)
    return model


if __name__ == "__main__":
    joined_df = load_data()
    processed_df = preprocess(joined_df)
    train(processed_df)