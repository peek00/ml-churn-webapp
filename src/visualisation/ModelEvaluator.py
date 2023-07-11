from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch.nn as nn
import torch
from torchvision import datasets
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from torchvision.transforms import ToPILImage
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from backbone.ResNet18_BinaryClassification import ResNet18_BinaryClassification
from DisplayUtility.display_utility import save_images
from classes.TensorboardLogger import TensorBoard_Logger
@dataclass
class ModelEvaluator:
    """
    ModelEvaluator was written to try to be a higer level class for evaluating trained models.
    Currently, the methods are catered towards a binary classification model and slightly hard coded.
    """
    dataloader: Optional[torch.utils.data.DataLoader] = None
    model: Optional[nn.Module] = None
    image_output_dir: Optional[Path] = None

    def __init__(self, name:str)->None:
        self.name = name
        self.tb_logger = TensorBoard_Logger(self.name)
        self.set_device()

    def check(self)->None:
        """
        Asserts that model and dataloader has been loaded, automatically ran before calling evaluate.
        """
        assert self.model is not None, "Model has not been loaded!"
        assert self.dataloader is not None, "Dataloader has not been loaded!"

    def set_image_output(self, dir:Path)->None:
        """
        Sets output directory for images to be saved.

        Args:
            dir: Path representing a folder. 
        """
        # Makes folder if does not exists
        os.makedirs(dir, exist_ok=True)
        self.image_output_dir = dir

    def load_model(self, model_class:nn.Module, model_state_dict:Path=None)->None:
        """
        If a state dictionary path is provided, load it into the model and set as self.
        Sends model to DEVICE as well.
        
        Args:
            model_class: An nn.Module representing the desired architecture
            model_state_dict: A string Path pointing to the state dictionary to be loaded, if any
        """
        model = model_class
        if model_state_dict:
            model.load_state_dict(torch.load(model_state_dict))
        model.to(self.DEVICE)
        self.model = model
        
    def set_device(self)->None:
        """
        Sets self.DEVICE to cuda if available; used in to(self.DEVICE) at required areas.
        """
        self.DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device set to {self.DEVICE}.")

    def check_device(self)->str:
        return self.DEVICE

    def set_seed(self, seed:int=666)->None:
        torch.manual_seed(seed)

    def set_dataloader(self, dataloader: torch.utils.data.DataLoader)->None:
        """
        Takes in an instantiated dataloader object to be used to evaluate on.
        
        Args:
            dataloader: An instantiated dataloader object
        """
        assert len(dataloader) > 0 and dataloader is not None, "Please provide a valid dataloader object."
        self.dataloader = dataloader

    def set_filenames(self, filenames):
        self.filenames = filenames

    @torch.no_grad()
    def evaluate(self)-> Tuple[List, List, List]:
        """
        Runs the model on self.dataloader and returns 3 lists.

        Returns:
            - list: Images in Tensor format
            - list: y_pred in logits / probability format
            - list: y_true labels in Tensor format

        """
        self.check()
        y_pred = []
        y_true = []
        images_list = []
        filenames = []
        for images, labels, paths in self.dataloader:
            images, labels = images.to(self.DEVICE), labels.to(self.DEVICE)

            outputs = self.model(images)
            for i, output in enumerate(outputs):
                y_pred.append(output.item())
                y_true.append(labels[i].item())
                images_list.append(images[i])
            filenames.extend(paths)
        return images_list, filenames, y_pred, y_true

    def get_fpr_tpr_threshold_auc(self, y_pred:List[float], y_true:List[int])-> Tuple[float, float]:
        """
        Given y_pred and y_true where y_pred is the probability of being part of the positive class,
        and y_true as the actual class (1,0), calculate and return the best threshold
        and auc using youden's method.

        Args:
            - y_pred: List containing the probability outputs from model.
            - y_true: List containing the true labels

        Returns:
            - float: Float value representing the best threshold that maximises Youden's score
            - auc: Float value representing the AUC calculated
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        youden = tpr - fpr
        best_threshold = thresholds[np.argmax(youden)]

        return fpr, tpr, thresholds, auc
    
    def log_incorrect_predictions(self,
                                      images: List[torch.Tensor],
                                      filenames: List[str],
                                      y_pred: List[float],
                                      y_true: List[int],
                                      threshold: float= 0.5,
                                      suffix:str=None,
                                      limit:int=50,
                                      )->None:
        assert len(images) == len(y_pred) == len(y_true), "Image list, y_pred and y_true are not of same length!"
        print("--------Log Incorrect Predictions------------")
        print(f"Using threshold of {threshold}.")
        print("--------Log Incorrect Predictions------------")
        binarized_pred = [1 if x > threshold else 0 for x in y_pred]

        false_negatives_images = []
        false_negatives_labels = []
        false_negatives_paths = []

        false_positives_images = []
        false_positives_labels = []
        false_positives_paths = []

        for i in range(len(images)):
            if binarized_pred[i] != y_true[i]:
                if y_true[i] == 1: # Means true label 1, predicted as 0, false negative
                    false_negatives_images.append(images[i])
                    false_negatives_labels.append(str(binarized_pred[i]))
                    false_negatives_paths.append(filenames[i])
                elif y_true[i] == 0:
                    false_positives_images.append(images[i])
                    false_positives_labels.append(str(binarized_pred[i]))
                    false_positives_paths.append(filenames[i])
        self.tb_logger.display_images("False Negative", 
                                      false_negatives_images,
                                      false_negatives_labels)
        self.tb_logger.display_images("False Positives", 
                                      false_positives_images,
                                      false_positives_labels)

    def save_incorrect_predictions(self,
                                      images: List[torch.Tensor],
                                      filenames: List[str],
                                      y_pred: List[float],
                                      y_true: List[int],
                                      threshold: float= 0.5,
                                      suffix:str=None,
                                      limit:int=50,
                                      )->None:
        """
        Uses DisplayUtility class (written by @peiduo) to save the incorrect predictions (both fn and fp) into png forms.

        Args:
            - images: List of torch tensors representing the images
            - y_pred: List containing the probability outputs from model.
            - y_true: List containing the true labels
            - threshold: Float where values above are binarized into 1s
            - dir: String representing path to save the image
            - suffix: String representing suffix
            - limit: (not implemented) Int capping how many images to display in an image 
        
        """
        assert len(images) == len(y_pred) == len(y_true), "Image list, y_pred and y_true are not of same length!"
        
        print(f"Using threshold of {threshold}.")
        binarized_pred = [1 if x > threshold else 0 for x in y_pred]

        false_negatives_images = []
        false_negatives_labels = []
        false_negatives_paths = []

        false_positives_images = []
        false_positives_labels = []
        false_positives_paths = []

        to_pil = ToPILImage()
        for i in range(len(images)):
            if binarized_pred[i] != y_true[i]:
                if y_true[i] == 1: # Means true label 1, predicted as 0, false negative
                    pil_image = to_pil(images[i])
                    false_negatives_images.append(pil_image)
                    false_negatives_labels.append(str(binarized_pred[i]))
                    false_negatives_paths.append(filenames[i])
                elif y_true[i] == 0:
                    pil_image = to_pil(images[i])
                    false_positives_images.append(pil_image)
                    false_positives_labels.append(str(binarized_pred[i]))
                    false_positives_paths.append(filenames[i])
        if suffix:
            fn_name = "false_negative_" + suffix
            fp_name = "false_positive_" + suffix
        else:
            fn_name = "false_negative"
            fp_name = "false_positive"
        save_images(false_negatives_images, false_negatives_paths, fn_name, self.image_output_dir, need_show_labels=True)
        save_images(false_positives_images, false_positives_paths, fp_name, self.image_output_dir, need_show_labels=True)

    def save_binary_confusion_matrix(self, 
                                     y_pred:List, 
                                     y_true:List, 
                                     threshold:float, 
                                     labels:List[str] = ['Negative', 'Positive'],
                                     name:str="confusion_matrix.png"
                                     )->None:
        """
        Saves an image of a 2 by 2 binary confusion matrix. 

        Args:
            - y_pred: List containing the probability outputs from model.
            - y_true: List containing the true labels
            - threshold: Float representing threshold for positive class
            - labels: List in the form ["Negative Class", "Positive Class"]
            - name: Str representing filename
        
        Returns:
            - tuple: Form of (tn, fp, fn, tp)
        """
        y_pred = np.array([1 if x > threshold else 0 for x in y_pred])
        y_true = np.array(y_true)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()
      
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figure size as needed
        
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Use plt.imshow instead of ax.imshow
        plt.colorbar(im, ax=ax)  # Use plt.colorbar instead of ax.figure.colorbar
        
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            xlabel='Predicted label', ylabel='True label',
            title='Confusion Matrix',
            aspect='auto')

        # Add text annotations to each cell
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()  # Adjust the margins for better padding

        if self.image_output_dir:
            name = os.path.join(self.image_output_dir, name)
        plt.savefig(name)
        return tn, fp, fn, tp
    
    def get_metrics_across_thresholds(self,
                                      y_pred:List[float],
                                      y_true:List[int],
                                      lower_bound:float=0.6, 
                                      upper_bound:float=0.9, 
                                      step:float=0.01):
        """
        Calculates accuracy, precision, recall and f1 scores given a range of threshold values bounded by lower and upper bound. 

        Args:
            - y_pred: List containing the probability outputs from model.
            - y_true: List containing the true labels
            - lower_bound: Float representing the lower bound of the threshold calculation.
            - upper_bound: Float representing the upper bound of the threshold calculation.
            - step: Float representing each increments.

        Returns:
            - Dictionary object in the following form. Threshold is saved as an int to 
              prevent any floating point issues at small step sizes.
                    threshold : {
                        factor:int, Divide threshold by this number to get actual threshold
                        accuracy:float,
                        precision:float,
                        recall:float,
                        f1:float,
                    }
        """
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
    
    def plot_pr_curve(self, y_pred, y_true):
        """
        **UNUSED**
        Uses TensorBoard SummaryWriter plot_pr_curve function.
        """
        self.tb_logger.plot_pr_curve("PR Curve", y_pred, y_true)
