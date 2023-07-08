from data_preprocessor import DataPreprocessor
from data_builder import DataBuilder
from data_etl import DataETL
from model.XGBoost import XGBoost
from model.CatBoost import CatBoost
from model.evaluator import ModelEvaluator
import pickle
from visualisation.PlotterUtility import PlotterUtility
from sklearn.metrics import confusion_matrix, roc_curve
from pathlib import Path
import yaml
import pandas as pd

def get_config(config:Path, model_type:str):
    with open(config, 'r') as file:
        config = yaml.safe_load(file)
    for model in config['models']:
        if model["model_type"] == model_type:
            return model

def load_data():
    etl = DataETL()
    # etl.use_local_data("research/data_given/")
    etl.retrieve_tables()
    etl.join_tables()
    return etl.get_df()

def preprocess(df:pd.DataFrame, target, features, encoders):
    dp = DataPreprocessor(df)
    dp.preprocess()
    preprocessed_data = dp.transform(target, features, encoders)
    return preprocessed_data

def train(df:pd.DataFrame, model):
    """
    Takes in the joined and preprocess full df.
    """
    databuilder = DataBuilder(df)
    databuilder.perform_pca(5)
    databuilder.save_pca()

    X_train = databuilder.get_transformed_X_train()
    y_train = databuilder.get_y_train()
    # features_list = ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]

    model.train(X_train, y_train)
    # return model, databuilder.get_X_train(), databuilder.get_y_train(), databuilder.get_X_test(), databuilder.get_y_test()
    return model, databuilder.get_transformed_X_train(), databuilder.get_y_train(), databuilder.get_transformed_X_test(), databuilder.get_y_test()

def evaluate(model, X_test:pd.DataFrame, y_test:pd.Series):

    metrics_dict, metrics_across_thresholds = ModelEvaluator(model, X_test, y_test)\
    .evaluate(ModelEvaluator.supported_metrics)
    for metric, value in metrics_dict.items():
        print(f"{metric} = {round(value, 4)}")
    return metrics_across_thresholds



if __name__ == "__main__":
    model_type = "catboost"
    config = get_config('src/config.yml', model_type)

    df = load_data()
    processed_df = preprocess(df, 
                              config['target'], 
                              config['features_needed'], 
                              config['encoders_needed']
                              )
    # Initialize model
    model = CatBoost("catboost")
    model, X_train, y_train, X_test, y_test = train(processed_df, model)

    # Loading model instead, override the model
    # model_path = "D:/GitHub/AI300_Capstone/team08/backend/model/catboost_model.pkl"
    # with open(model_path, 'rb') as file:
    #     model = pickle.load(file)
    metrics_across_thresholds = evaluate(model, X_test, y_test)

    # Visualiing
    # plotter = PlotterUtility("catboost", mode="plotly")
    # plotter.plot_metrics_across_thresholds(metrics_across_thresholds,  focus="accuracy")
    
    # y_pred_prob = model.predict_proba(X_test)
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    # plotter.plot_roc_curve(fpr, tpr, thresholds)

    
    # model.save()
