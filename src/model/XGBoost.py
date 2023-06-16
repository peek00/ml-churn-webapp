from abc import ABC, abstractmethod
import xgboost as xgb
import joblib
from sklearn.model_selection import KFold
import numpy as np

class Model(ABC):
    '''
      Interface used for CatBoostModel.
      This class serves the purposes of
      (1) Documentation, if your teammates wish to understand how to use it
      (2) Extending this interface or "template" to other models
    '''
    @abstractmethod
    def train(self, X, y):
        '''Train model with features and labels'''
        pass

    @abstractmethod
    def predict(self, X):
        '''Generate binary predictions (0/1) from trained model'''
        pass

    @abstractmethod
    def predict_proba(self, X):
        '''Generate prediction probabilities from trained model'''
        pass

    @abstractmethod
    def load(self, input_file_path):
        '''Load trained model from file'''
        return joblib.load(input_file_path)

    @abstractmethod
    def save(self, model, output_file_path):
        '''Save trained model to file'''
        joblib.dump(model, output_file_path)

class XGBoost(Model):
    def __init__(self):
        self.model = xgb.XGBClassifier()
    
    def train(self, X, y, n_splits=5):
        y = y.astype(int)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]  # Updated indexing
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]  # Updated indexing
            print(f"Y_train is {y_train.value_counts()}")
            print(f"Null for y_train is {y_train.isna().sum()}")
            print(f"Y_val is {y_val.value_counts()}")
            print(f"Null for y_val is {y_val.isna().sum()}")
            self.model.fit(X_train, y_train)

            # Evaluate the model on the validation set
            val_score = self.model.score(X_val, y_val)
            scores.append(val_score)
        
        avg_score = np.mean(scores)
        print(f"Average validation score: {avg_score}")

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
    
    def load(self):
        pass

    def save(self):
        pass
    

if __name__ == "__main__":
    etl = DataETL()
    etl.use_local_data("research/data_given/")
    etl.join_tables()

    df = etl.get_df()
    dp = DataPreprocessor(df)

    db = DataBuilder(dp.get_df())
    print(db.get_transformed_X_train().head())
    
    pass