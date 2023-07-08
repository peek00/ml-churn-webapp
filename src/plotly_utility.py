from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pathlib import Path
import os
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import auc

@dataclass
class PlotterUtility:
    """
    Utility class to plot graphs. Uses Matplotlib and Plotly to plot certain graphs. 
    Certain functions may not have been implemented in both modes currently.
    """
    run_name:str
    output_dir:Path="plots/single_plots"
    fig_size:Tuple[int]=(16,9)
    mode:str="matplotlib"
    width:int=1280
    height:int=720

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_roc_curve(self,
                       fpr:List[float],
                       tpr:List[float],
                       thresholds:List[float]=None,
                       x_ticks:List[float]=[1e-3, 1e-2, 1e-1, 0.5, 1],
                       )->None:
        """
        Plots and saves a ROC Curve with the run prefix using either Matploblib or Plotly.
        
        Args:
            - fpr: List of float values representing the fpr
            - tpr: List of float values representing the tpr
            - thresholds: List of float values representing the thresholds
            - x_ticksL List of float values representing desired values to be plotted
            FPR, TPR and Thresholds are obtained from sklearns roc_curve function.
        """
        if self.mode == "matplotlib":
            self.__matplot_plot_roc_curve(fpr, tpr, thresholds, x_ticks)
        elif self.mode == "plotly":
            self.__plotly_plot_roc_curve(fpr, tpr, thresholds, x_ticks)

    def plot_metrics_across_thresholds(self,
                                       metrics_obj:Dict,
                                       focus:str="accuracy"
                                       )->None:
        """
        Plots and saves the different metrics across thresholds. Only implemented in Plotly currently.
        
        Args:
            - metrics_obj: Dict in the format 
                    threshold : {
                        factor:int, Divide threshold by this number to get actual threshold
                        accuracy:float,
                        precision:float,
                        recall:float,
                        f1:float,
                    }
                * metrics_obj can be obtained from ModelEvaluator.get_metrics_across_thresholds()
            - focus:str, Represents which metric to plot on the y-axis.
        """
        assert focus in ['accuracy', 'precision', 'recall', 'f1'], "Focus must be one of accuracy, precision, recall or f1!"
       
        if self.mode == "matplotlib":
            raise NotImplementedError
        elif self.mode == "plotly":
            self.__plotly_plot_metrics_across_thresholds(metrics_obj, focus=focus)

    def __plotly_plot_metrics_across_thresholds(self,
                                                metrics_obj:Dict,
                                                focus:str="accuracy"
                                                )->None:
        """
        Uses Plotly to plot a scatter graph of chosen focus over the thresholds.
        Currently highlights points that have the highest accuracy, precision, recall or f1 score. If a point is the highest in two or more metric, only one color will be highlighted.
        Legend is implemented in a custom manner: They are NOT linked to the actual traces.
        """
        max_acc = (0, -1) # Value, index
        max_precision = (0, -1) # Value, index
        max_f1 = (0, -1) # Value, index
        max_recall = (0, -1) # Value, index

        thresholds = []
        focus_metric = []
        tooltips = []
        for i, (threshold,metrics) in enumerate(metrics_obj.items()):
            actual_threshold = threshold / metrics['factor']
            thresholds.append(actual_threshold)
            focus_metric.append(metrics[focus])

            tooltip_text = "<b>Metrics:<b></br>"
            tooltip_text += f"threshold:{actual_threshold}<br>"
            for k,v in metrics.items():
                if k == "factor":
                    continue
                tooltip_text += f"{k}:{v}<br>"
                if k == "accuracy":
                    max_acc = max(max_acc, (v,i), key=lambda x:x[0])
                elif k == "precision":
                    max_precision = max(max_precision, (v,i), key=lambda x:x[0])
                elif k == "recall":
                    max_recall = max(max_recall, (v,i), key=lambda x:x[0])
                elif k == "f1":
                    max_f1 = max(max_f1, (v,i), key=lambda x:x[0])

            tooltips.append(tooltip_text)

        # Plotting
        fig = px.line(
            x=thresholds,
            y=focus_metric,
            hover_data={"Tooltip": tooltips},
            labels=dict(x="Thresholds", y=focus),
            markers=True,
            width=self.width, height=self.height
        )

        # Setting title
        fig.update_layout(
            title=dict(
            text=f"{self.run_name} Metrics across Thresholds | {focus.capitalize()}",
            font=dict(size=20),
            automargin=True,
            xanchor="center", #Centers title text
            x=0.5,
            pad=dict(t=20) # Gives top padding
            ),
        )

        # Rotating x-axis labels to be more readable
        fig.update_layout(xaxis={
            'type': 'category',
            'tickangle':45
            })

        # Setting hover data
        if tooltips is not None:
            hovertemplate = '<b>%{customdata}</b><extra></extra>'
            fig.data[0].customdata = tooltips
            fig.data[0].hovertemplate = hovertemplate

        # Getting markers with highest x
        indexes = [max_f1[1], max_precision[1], max_recall[1], max_acc[1]]
        names = ["Highest F1", "Highest Precision", "Highest Recall", "Highest Accuracy"]

        highlighted_thresholds =  [thresholds[i] for i in indexes]
        highlighted_focus_metric =  [focus_metric[i] for i in indexes]
        highlighted_tooltips = [tooltips[i] for i in indexes]
        # Assigning colors to markers
        colors = ["green", "orange", "purple", "red"]

        # Adding specific markers
        fig.add_trace(
            go.Scatter(
                x=highlighted_thresholds,
                y=highlighted_focus_metric,
                hovertext=highlighted_tooltips,
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=14,
                    color=colors
                ),
                showlegend=False
            )
        )

        # Add dummy traces with custom names for the legend
        for i, color in enumerate(colors):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color=color
                    ),
                    name=names[i]
                )
            )

        # Update the layout to show the legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
            x=0.85,
            y=1,
            bgcolor="rgba(0,0,0,0)",  # Set the background color of the legend to transparent
            bordercolor="rgba(0,0,0,0)"  # Set the border color of the legend to transparent
            ) 
        )

        # Saving file
        file_name = f"{self.run_name}_Metrics_across_Thresholds.html"
        path = os.path.join(self.output_dir, file_name)
        # assert os.path.exists(path) == False, f"Plot {path} already exists."
        fig.write_html(path, include_plotlyjs="directory")
                
    def __matplot_plot_roc_curve(self, 
                       fpr:List[float], 
                       tpr:List[float],
                       thresholds:List[float],
                       x_ticks:List[float]=[1e-3, 1e-2, 1e-1, 0.5, 1],
                       )->None:
        # Plot the ROC curve
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=self.fig_size)  # Set figure size to 1920x1080

        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'{self.run_name} (ROC) Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xscale("log")
        plt.xticks(ticks=x_ticks, labels=[str(x) for x in x_ticks])

        file_name = f"{self.run_name}_ROC_Curve.png"
        path = os.path.join(self.output_dir, file_name)

        assert os.path.exists(path) == False, f"Plot {path} already exists."
        
        plt.savefig(path)

    def __plotly_plot_roc_curve(self,
                        fpr:List[float], 
                        tpr:List[float],
                        thresholds:List[float],
                        x_ticks:List[float]=[1e-3, 1e-2, 1e-1, 0.5, 1]
                        )->None:
        # Create the ROC curve trace
        fig = px.area(
            x=fpr, y=tpr,
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=self.width, height=self.height
        )

        # Setting title
        fig.update_layout(
            title=dict(
            text=f"{self.run_name} ROC Curve | (AUC={auc(fpr, tpr):.4f})",
            font=dict(size=20),
            automargin=True,
            xanchor="center", #Centers title text
            x=0.5,
            pad=dict(t=20) # Gives top padding
            ),
        )

        # Set x-axis to log and add custom values
        fig.update_xaxes(type='log')
        fig.update_xaxes(tickvals=x_ticks)

        # Add thresholds to the hovertemplate
        if thresholds is not None:
            hovertemplate = 'Threshold: %{text}<br>FPR: %{x}<br>TPR: %{y}<extra></extra>'
            fig.data[0].text = [f'{threshold:.4f}' for threshold in thresholds]
            fig.data[0].hovertemplate = hovertemplate

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')

        # Saving file
        file_name = f"{self.run_name}_ROC_Curve.html"
        path = os.path.join(self.output_dir, file_name)
        assert os.path.exists(path) == False, f"Plot {path} already exists."
        fig.write_html(path, include_plotlyjs="directory")
