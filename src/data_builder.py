from data_etl import DataETL

import pandas as pd
from sklearn.model_selection import train_test_split
class DataBuilder:
    """
    This code takes in the preprocessed and joined dataframe and is responsible for:
    - Dropping target column 
    - Creating train - test split
    - Performing PCA to get top 5 features
    """

    def __init__(self, df: pd.DataFrame):
        """
        Takes in the joined DF. 
        """
        self.df = df
        self.build_train_test_set()


    def build_train_test_set(self):
        """
        Splits train and test set using stratified train-test split
        """
        X = self.df.drop('churn_label', axis=1)
        y = self.df['churn_label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        class_counts = y_train.value_counts()
        class_imbalance_ratio = class_counts[1] / class_counts[0]
        print("Class Imbalance Ratio in Training Set:", class_imbalance_ratio)

if __name__ == "__main__":
    etl = DataETL()
    etl.use_local_data("research/data_given/")
    etl.join_tables()

    db = DataBuilder(etl.get_df())