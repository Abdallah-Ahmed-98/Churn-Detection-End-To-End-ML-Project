import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc
import numpy as np
import joblib
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):

        # Classification metrics
        accuracy = round(accuracy_score(actual, pred), 2)
        f1 = round(f1_score(actual, pred, average="weighted"), 2)
        recall = round(recall_score(actual, pred, average="weighted"), 2)
        precision = round(precision_score(actual, pred, average="weighted"), 2)
        
        # Confusion matrix
        cm = confusion_matrix(actual, pred)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(actual, pred)
        roc_auc = auc(fpr, tpr)


        return accuracy, f1, recall, precision, cm, fpr, tpr, roc_auc


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (accuracy, f1, recall, precision, cm, fpr, tpr, roc_auc) = self.eval_metrics(test_y, predicted_qualities)
        

        ## -------- Confusion Matrix ------------------ ##
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.savefig(f'{self.config.confusion_matrix_file_path}', bbox_inches='tight', dpi=300)
        plt.close()

        ## -------- ROC Curve ------------------ ##
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.config.roc_curve_file_path}')  
        plt.close()
        
        # Saving metrics 
        scores = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
        save_json(path=Path(self.config.metric_file_name), data=scores)