from sklearn.metrics import roc_auc_score, precision_score, \
    recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List
from collections import defaultdict

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
                y_pred = self.model.predict_proba(self.X) # Need to transform self.x into pca
                y_pred_prob = np.array(y_pred) # Assuming y_pred_prob has shape (n_samples, n_classes)
               
                auc, threshold = self.evaluate_auc_threshold(self.y, y_pred_prob)
                y_pred = self._convert_to_binary(y_pred, threshold)
            else:
                y_pred = self.model.predict(self.X)
            metrics_dict[metric] = self._calculate(metric, self.y, y_pred)

        # For extra visulization
        metrics_across_thresholds = self.get_metrics_across_thresholds(y_pred_prob, self.y)
        return metrics_dict, metrics_across_thresholds
    
    def get_metrics_across_thresholds(self,
                                    y_pred:List[float],
                                    y_true:List[int],
                                    lower_bound:float=0.50, 
                                    upper_bound:float=0.95, 
                                    step:float=0.01
                                    ):
        metrics = defaultdict(dict)
        # Converts the lower and upper bound into int characters, maintaining precision so the for loop works
        factor = 10 ** len(str(step).split('.')[-1])  # Determine the factor based on the decimal places in the step value

        lower_bound_int = int(lower_bound * factor)  # Convert float to integer with precision preservation
        upper_bound_int = int(upper_bound * factor)  # Convert float to integer with precision preservation
        step_int = int(step * factor)  # Convert float to integer with precision preservation

        for threshold in range(lower_bound_int, upper_bound_int, step_int):
            # Converts back to the intended value with precision preserved
            threshold_float = threshold / factor  
            # Use the integer threshold as keys / slightly worried about floating point bullshit
            # Factor is stored to divide by
            metrics[threshold] = self.calculate_metrics(y_pred, y_true, threshold_float)
            metrics[threshold]["factor"] = factor
    
        return metrics
    
    def calculate_metrics(self,
                        y_pred:List, 
                        y_true:List,
                        threshold:int=0.5
                        )->List[float]:
        """
        Calculates accuracy, precision, recall and f1 score based on threshold and returns.

        Args:
            - y_pred: List containing the probability outputs from model.
            - y_true: List containing the true labels
            - threshold: Float representing threshold for positive class
        """
        y_pred = np.array([1 if x > threshold else 0 for x in y_pred])
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / (1 / precision + 1 / recall)

        return {
            "accuracy" : round(accuracy.item(), 3),
            "precision" : round(precision.item(), 3),
            "recall" : round(recall.item(), 3),
            "f1" : round(f1.item(), 3)
        }
    
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
