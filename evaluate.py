import os
import numpy as np
import pickle 
import argparse 
import matplotlib.pyplot as plt 
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report, plot_precision_recall_curve

def load_data(feature_file, label_file):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y

def load_checkpoint(input_dir, input_file):
    with open(os.path.join(input_dir, input_file), 'rb') as f:
        return pickle.load(f)
    

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    target_names = ['No', 'Yes']
    # print(confusion_matrix(y_test, y_pred, labels=target_names))
    print(classification_report(y_test, y_pred, target_names=target_names))

def plot_result(model, X_test, y_test):
    disp = plot_precision_recall_curve(model, X_test, y_test)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', type=str)
    parser.add_argument("--label_file", type=str)
    parser.add_argument("--do_plot_result", action="store_true")
    parser.add_argument("--checkpoint_dir", default="saved_models", type=str)
    parser.add_argument("--checkpoint_file", type=str)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint_dir, args.checkpoint_file)
    X_test, y_test = load_data(args.feature_file, args.label_file)
    evaluate(model, X_test, y_test)
    if args.do_plot_result:
        plot_result(model, X_test, y_test)