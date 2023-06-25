from sklearn.metrics import roc_auc_score, precision_score, \
    recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np


class ModelEvaluator:
    '''
        Model-agnostic & metric-agnostic evaluator.
        Currently supports AUC, precision, recall, and F1 score
    '''
    supported_metrics = ['auc', 'precision', 'recall', 'f1']

    def __init__(self, model, X, y, feature_list=None):
        self.model = model
        if feature_list is not None:
            self.X = X[feature_list]
        else:
            self.X = X
        self.y = y.astype(int)


    def evaluate_auc_threshold(self, y_true, y_pred_prob):
    # Calculate FPR, TPR, and thresholds
    
        y_true = y_true.to_numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        # Convert y_true to nd.array
        # Calculate AUC
        auc = roc_auc_score(y_true, y_pred_prob)
        # Find the threshold that maximizes the trade-off between TPR and FPR
        optimal_threshold_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_idx]

        return auc, optimal_threshold
    
    def evaluate(self, metric_list):

        metrics_dict = {}
        for metric in metric_list:
            if metric == 'auc':
                y_pred = self.model.predict_proba(self.X)
                y_pred_prob = np.array(y_pred) # Assuming y_pred_prob has shape (n_samples, n_classes)
                auc, threshold = self.evaluate_auc_threshold(self.y, y_pred_prob)
                y_pred = self._convert_to_binary(y_pred, threshold)
            else:
                y_pred = self.model.predict(self.X)
            metrics_dict[metric] = self._calculate(metric, self.y, y_pred)

        return metrics_dict
    
    def _convert_to_binary(self, y_pred, threshold):
        return (y_pred >= threshold).astype(int)

    def _calculate(self, metric, true_labels, pred_labels, threshold:float=0.7):
        assert metric in self.supported_metrics
        metric_map = {
            'auc': roc_auc_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score
        }
        # To minimize code repetition, use dict to map metric to sklearn fn
        return metric_map[metric](true_labels, pred_labels)
