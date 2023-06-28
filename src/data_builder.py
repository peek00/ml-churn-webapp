from data_etl import DataETL
from data_preprocessor import DataPreprocessor
import pickle
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from typing import List

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
        self.pca = None
        self.df = df
        self.build_train_test_set()
        self.get_corr_to_target()
        # self.perform_pca()
        self.save()


    def build_train_test_set(self):
        """
        Splits train and test set using stratified train-test split
        """

        X = self.df.drop('churn_label', axis=1)
        y = self.df['churn_label']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True).astype(int)
        self.y_test = self.y_test.reset_index(drop=True).astype(int)
        class_counts = self.y_train.value_counts()

        # print(f"Class counts is: {class_counts}")
    def get_X_train(self)->pd.DataFrame:
        return self.X_train
    
    def get_X_test(self)->pd.DataFrame:
        return self.X_test

    def get_corr_to_target(self):
        """
        Prints the correlation of each feature to the target.
        """
        # Get the correlation of each feature to the target
        corr_with_target = self.X_train.corrwith(self.y_train, numeric_only=True)
        abs_corr_with_target = corr_with_target.abs().sort_values(ascending=False)

        top_positive_features = [(feature, corr_with_target[feature]) for feature in abs_corr_with_target[corr_with_target > 0].index]
        top_negative_features = [(feature, corr_with_target[feature]) for feature in abs_corr_with_target[corr_with_target < 0].index]

        print("Top positively correlated features:")
        for feature, corr_score in top_positive_features:
            print(f"{feature}: {corr_score}")
        print("----------------------------------")
        print("Top negatively correlated features:")
        for feature, corr_score in top_negative_features:
            print(f"{feature}: {corr_score}")

        # Trying ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]

    def perform_pca(self, n_components:int=5)->pd.DataFrame:
        """
        Performs PCA on just the X_train and returns the top 5 features.
        """
        features = self.X_train.drop(columns=['customer_status', 'customer_id', 'account_id', 'zip_code'])  # Replace 'target' with your churn_label variable column name
        # print(f"I need this order: {features.columns}")
        
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(features)
        self.transformed_X_train = pd.DataFrame(data=transformed_features, columns=[f"PC{i+1}" for i in range(n_components)])

        # Retrieving important features
        # get the index of the most important feature on EACH component
        # LIST COMPREHENSION HERE
        most_important = [np.abs(self.pca.components_[i]).argmax() for i in range(n_components)]

        initial_feature_names = features.columns
        # get the names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_components)]

        # LIST COMPREHENSION HERE AGAIN
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_components)}

        # build the dataframe
        df = pd.DataFrame(dic.items())
        print(f"The top {n_components} features are: {df}")

        return self.transformed_X_train
    
    def get_transformed_X_train(self)->pd.DataFrame:
        return self.transformed_X_train
    
    def get_y_train(self)->pd.Series:
        return self.y_train
    
    def get_transformed_X_test(self)->pd.DataFrame:
        # Apply PCA using 
        features = self.X_test.drop(columns=['customer_status', 'customer_id', 'account_id', 'zip_code'])  # Replace 'target' with your churn_label variable column name
        self.transformed_X_test = self.pca.transform(features)
        return self.transformed_X_test
    
    def get_y_test(self)->pd.Series:
        return self.y_test
    
    def save(self):
        """
        Saves pca object to pickle file.
        """
        if self.pca != None:
            with open('pca.pkl', 'wb') as file:
                pickle.dump(self.pca, file)
            print("Saved PCA object to pickle file.")

    def get_negative(self):
        # Filter the transformed X_train based on churn_label = 0
        churn_0_indices = self.y_train[self.y_train == 0].index
        print(churn_0_indices)
        # churn_0_rows = self.transformed_X_train.iloc[churn_0_indices].head(5)
        churn_0_rows = self.X_train.iloc[churn_0_indices].head(5)
        
        # Print the churn_label = 0 rows
        print(churn_0_rows)


if __name__ == "__main__":
    etl = DataETL()
    # etl.use_local_data("research/data_given/")
    etl.retrieve_tables()
    etl.join_tables()

    df = etl.get_df()
    print(df[df['churn_label'] == "Yes"][["tenure_months",'num_referrals','total_long_distance_fee','total_charges_quarter','contract_type']].head())
    print(df[df['churn_label'] == "No"][["tenure_months",'num_referrals','total_long_distance_fee','total_charges_quarter','contract_type']].head())
    # dp = DataPreprocessor(df)
    # db = DataBuilder(dp.get_df())
    # db.get_negative()