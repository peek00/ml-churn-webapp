from abc import ABC, abstractmethod

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