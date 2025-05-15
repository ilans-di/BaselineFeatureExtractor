#!/usr/bin/env python

import os

import pandas as pd
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder



from xgboost import XGBClassifier
import joblib
from feature_extractor import URLFeatureExtractor
from dataset_locator import TrainDS

def create_model(model_type, seed):
    match model_type:
        case 'svm':
            return SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
        case 'random_forest':
            return RandomForestClassifier(n_jobs=8,random_state=seed, verbose=1,
                                          n_estimators=100,
                                          max_depth=15,  # limit depth
                                          min_samples_leaf=5,  # don't grow for tiny groups
                                          max_features='sqrt',  # limit how many features are considered per split
                                          )

        case 'xgboost':
            return XGBClassifier(n_jobs=-1,use_label_encoder=False, eval_metric='logloss', random_state=seed, verbosity=1)
        case 'logistic_regression':
            return LogisticRegression(max_iter=10000, random_state=seed, verbose=1)
        case 'gradient_boosting':
            return GradientBoostingClassifier(random_state=seed, verbose=1)
        case 'extra_trees':
            return ExtraTreesClassifier(n_jobs=-1,random_state=seed, verbose=1)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='Model type')
    parser.add_argument('--dataset_name', type=str, required=True, help='The name of the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()

# Load datasets
    train = pd.read_parquet(TrainDS[args.dataset_name].value)
    print(f"Dataset {args.dataset_name} loaded")

    # Feature extraction
    extractor = URLFeatureExtractor()
    extractor.fit(train)

    print(f"entropy features extracted")


    X_train = extractor.transform(train)
    print(f"Features extracted")

    # Relabel
    if args.dataset_name == "deepurlbench":
        train['label_binary'] = train['label'].map(lambda x: 'benign' if x == 'benign' else 'malicious')

        # Labels
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train['label_binary'])
    else:
        y_train = train['label']!="benign"
        label_encoder = None
    # Choose model
    model = create_model(args.model_type, args.seed)
    print(f"Starting training {args.model_type}")

    model.fit(X_train, y_train)
    print(f"{args.model_type} trained")

    # Save
    model_name = f"{args.model_type}_{args.dataset_name}_{args.seed}"

    joblib.dump({'extractor': extractor, 'model': model, 'label_encoder': label_encoder},
                os.path.join("models", f"{model_name}.pkl"))
