#!/usr/bin/env python
import os

import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
#from dotenv import load_dotenv
from clearml import Task
from dataset_locator import TestDS #it is an enum for path of the dataset
import joblib

def recall_at_fpr(y_true, y_probs, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Find indices where FPR is smaller or equal to target
    valid = np.where(fpr <= target_fpr)[0]

    if len(valid) == 0:
        return 0.0  # No threshold achieves such a low FPR
    else:
        best_idx = valid[-1]  # Take the highest threshold that still meets FPR <= target
        return tpr[best_idx]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to saved model file')
parser.add_argument('--dataset_name', type=str, required=True, help='The name of the dataset')
args = parser.parse_args()
data = joblib.load(args.model)

print(f"Model {args.model} loaded")
extractor = data['extractor']
clf = data['model']
label_encoder = data['label_encoder']

test = pd.read_parquet(TestDS[args.dataset_name].value)
print(f"Dataset {args.dataset_name} loaded")
X_test = extractor.transform(test)
if clf.n_classes_ ==2:
    y_test_binary = test["label"] !="benign"
else:

    y_test = label_encoder.transform(test['label'])
    print(f"Transformation completed")

    # Inverse transform to get string labels
    y_test_str = label_encoder.inverse_transform(y_test)

    # Create binary ground truth
    y_test_binary = np.array([0 if label == 'benign' else 1 for label in y_test_str])


# Concatenate the predictions
test_probs = clf.predict_proba(X_test)
if clf.n_classes_ ==2:
    malicious_probs = test_probs[:,1]
    print("using binary classifier")
else:
    # Find index for 'benign' class
    benign_class_idx = list(label_encoder.classes_).index('benign')
    
    # Maliciousness probability = 1 - benign probability
    malicious_probs = 1.0 - test_probs[:, benign_class_idx]
    print("using multiclass classifier")

    
test_auc = roc_auc_score(y_test_binary, malicious_probs)
task = Task.init(
            project_name=f'AlgoTeam/BaselineFeature/{args.dataset_name}',  # project name of at least 3 characters
            task_name=os.path.splitext(os.path.split(args.model)[1])[0],  # task name of at least 3 characters
            task_type="testing",
            tags=None,
            auto_connect_arg_parser=True,
            auto_connect_frameworks=False,  # This can either be a boolean, or a dictionary to select which hooks to use
            auto_resource_monitoring=True,
            auto_connect_streams=True,
            reuse_last_task_id=False,
        )
recall_fpr_1 = recall_at_fpr(y_test_binary, malicious_probs, target_fpr=0.01)
recall_fpr_01 = recall_at_fpr(y_test_binary, malicious_probs, target_fpr=0.001)
task.get_logger().report_scalar("AUC", "test", iteration=0, value=test_auc)
task.get_logger().report_scalar("Recall@0.1%", "test", iteration=0,
                                value=recall_fpr_01)
task.get_logger().report_scalar("Recall@1%", "test", iteration=0,
                                value=recall_fpr_1)

task.close()

print(f"Train AUC: {test_auc:.4f}")
print(f"Recall at 1% FPR: {recall_fpr_1:.4f}")

