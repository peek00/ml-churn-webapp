from data_etl import DataETL
from data_preprocessor import DataPreprocessor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
        pd.set_option('display.max_columns', None)  # None will display all columns
        self.df = df
        self.build_train_test_set()
        self.perform_pca()
        print(self.transformed_X_train.head())


    def build_train_test_set(self):
        """
        Splits train and test set using stratified train-test split
        """
        X = self.df.drop('churn_label', axis=1)
        y = self.df['churn_label']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        class_counts = self.y_train.value_counts()
        print("Number of churned: ", class_counts[1])
        print("Number of not churned: ", class_counts[0])

    def perform_pca(self, n_components:int=5)->pd.DataFrame:
        """
        Performs PCA on just the X_train and returns the top 5 features.
        """
        features = self.X_train.drop(columns=['status', 'customer_id', 'account_id', 'zip_code'])  # Replace 'target' with your churn_label variable column name
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(features)
        self.transformed_X_train = pd.DataFrame(data=transformed_features, columns=[f"PC{i+1}" for i in range(n_components)])
        return self.transformed_X_train
    
    def get_transformed_X_train(self)->pd.DataFrame:
        return self.transformed_X_train
    
    def get_y_train(self)->pd.Series:
        return self.y_train
    
    def get_transformed_X_test(self)->pd.DataFrame:
        # Apply PCA using 
        features = self.X_test.drop(columns=['status', 'customer_id', 'account_id', 'zip_code'])  # Replace 'target' with your churn_label variable column name
        self.transformed_X_test = self.pca.transform(features)
        return self.transformed_X_test
    
    def get_y_test(self)->pd.Series:
        return self.y_test

if __name__ == "__main__":
    etl = DataETL()
    etl.use_local_data("research/data_given/")
    etl.join_tables()

    df = etl.get_df()
    dp = DataPreprocessor(df)

    db = DataBuilder(dp.get_df())
    