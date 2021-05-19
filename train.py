from pandas.io import parsers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import os
import pickle
import argparse

def load_data(feature_file, label_file):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y
     

def train(model, X_train, y_train):
    _ = model.fit(X_train, y_train)
    print("Training score: {}".format(model.score(X_train, y_train)))
    return model
def save_model(model, filename, output_dir):
    with open(os.path.join(output_dir, filename), 'wb') as f:
        print('Saving model to {}'.format(os.path.join(output_dir, filename)))
        pickle.dump(model, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name to train")
    parser.add_argument("--feature_file", type=str,
                        help="The input feature file name where data is loaded")
    parser.add_argument("--label_file", type=str,
                        help="The input label file name where label is loaded")              
    parser.add_argument("--kernel", default="linear", type=str, help="kernel option for SVM")
    parser.add_argument("--C", default=1.0, type=float, help="Regularization term")
    parser.add_argument("--gamma", default="scale", help="Gamma value for rbf, poly and sigmoid kernel")
    parser.add_argument("--degree", default="3", type=int, help="Degree value for poly SVM")
    parser.add_argument("--output_file", default="saved_model.pkl", type=str)
    parser.add_argument("--output_dir", default="saved_models", type=str,
                        help="The output directory where the model will be written.")
    args = parser.parse_args()

    X_train, y_train = load_data(args.feature_file, args.label_file)
    model_name = args.model_name
    C = args.C
    if model_name == 'logistic': 
        model = LogisticRegression(C=C, max_iter=10000, random_state=0)
    elif model_name == 'svm':
        kernel = args.kernel 
        gamma = args.gamma 
        degree = args.degree
        if kernel == 'linear':
            model = LinearSVC(C = C)
        else: 
            model = SVC(kernel=kernel,C=C, gamma=gamma, degree=degree, random_state=0)
    else:
        model = DecisionTreeClassifier( random_state=0)
    _ = model.fit(X_train, y_train)
    output_file = args.output_file
    output_dir = args.output_dir
    save_model(model, output_file, output_dir)


