from data_preprocessor import DataPreprocessor
from data_builder import DataBuilder
from data_etl import DataETL
from model.XGBoost import XGBoost
from model.CatBoost import CatBoost
from model.evaluator import ModelEvaluator

import pandas as pd

def load_data():
    etl = DataETL()
    # etl.use_local_data("research/data_given/")
    etl.retrieve_tables()
    etl.join_tables()
    return etl.get_df()

def preprocess(df:pd.DataFrame):
    dp = DataPreprocessor(df)
    return dp.get_df()

def train(df:pd.DataFrame, model):
    """
    Takes in the joined and preprocess full df.
    """
    databuilder = DataBuilder(df)
    # Uncomment next two if using PCA
    # X_train = databuilder.get_transformed_X_train()
    # y_train = databuilder.get_y_train()

    X_train = databuilder.get_X_train()
    y_train = databuilder.get_y_train()
    features_list = ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]

    model.train(X_train, y_train, features_list)
    return model, databuilder.get_X_train(), databuilder.get_y_train(), databuilder.get_X_test(), databuilder.get_y_test()
    # return model, databuilder.get_transformed_X_train(), databuilder.get_y_train(), databuilder.get_transformed_X_test(), databuilder.get_y_test()

def evaluate(model, X_test:pd.DataFrame, y_test:pd.Series):
    features_list = ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]

    metrics_dict = ModelEvaluator(model, X_test, y_test, features_list)\
    .evaluate(ModelEvaluator.supported_metrics)
    for metric, value in metrics_dict.items():
        print(f"{metric} = {round(value, 4)}")


if __name__ == "__main__":
    joined_df = load_data()
    processed_df = preprocess(joined_df)

    model = CatBoost("catboost")

    model, X_train, y_train, X_test, y_test = train(processed_df, model)
    evaluate(model, X_test, y_test)
    model.save()
