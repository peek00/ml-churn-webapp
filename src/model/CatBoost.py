import catboost as cb
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import numpy as np
import pickle

from model.model_abc import Model
from data_etl import DataETL
from data_preprocessor import DataPreprocessor
from data_builder import DataBuilder

class CatBoost(Model):
    def __init__(self, name:str):
        self.model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=True, random_seed=42)
        self.name = name

    def train(self, X, y, n_splits=5):
        y = y.astype(int)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        count = 1

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]  # Updated indexing
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]  # Updated indexing
            self.model.fit(X_train, y_train)

            # Evaluate the model on the validation set
            val_score = self.model.score(X_val, y_val)
            scores.append(val_score)
            print(f"Validation score for split {count}: {val_score}")
            count += 1
        
        avg_score = np.mean(scores)
        print(f"Average validation score: {avg_score}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
    
    def load(self):
        with open('model.pkl', 'rb') as file:
            self.model = pickle.load(file)
            print("Model loaded successfully")

    def save(self):
        with open(f'{self.name}_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
            print(f"Model saved successfully to {self.name}_model.pkl")
if __name__ == "__main__":
    etl = DataETL()
    etl.retrieve_tables()
    etl.join_tables()

    df = etl.get_df()
    dp = DataPreprocessor(df)

    db = DataBuilder(dp.get_df())
    features_list = ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]

    model = CatBoost()
    X_train = db.get_X_train()
    y_train = db.get_y_train()
    model.train(X_train, y_train, features_list)
    model.train()

    pass