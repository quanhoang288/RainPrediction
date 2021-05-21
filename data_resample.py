import os
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
import argparse
import numpy as np

def random_oversample(X_train, y_train, ratio=1):
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    return ros.fit_sample(X_train, y_train)
def SMOTE_oversample(X_train, y_train, ratio=1):
    smote = SMOTE(sampling_strategy=ratio)
    return smote.fit_sample(X_train, y_train)
def random_undersample(X_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    return rus.fit_sample(X_train, y_train)
def tomek_undersample(X_train, y_train):
    tomek = TomekLinks()
    return tomek.fit_resample(X_train, y_train)
def enn_undersample(X_train, y_train):
    enn = EditedNearestNeighbours()
    return enn.fit_resample(X_train, y_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_strategy', type=str, required=True, help="Resampling method")
    parser.add_argument('--ratio', type=float, default=1.0, help="Sampling ratio (ratio after resampled)")
    parser.add_argument('--feature_file', type=str, required=True, help="Input feature file")
    parser.add_argument('--label_file', type=str, required=True, help="Input label file")
    parser.add_argument('--output_dir', type=str, default='data', help="Directory to save data")
    parser.add_argument('--output_feature_file', type=str, default='X_resampled.npy', help="Name of output feature file")
    parser.add_argument('--output_label_file', type=str, default='y_resampled.npy', help="Name of output label file")
    args = parser.parse_args()
    X = np.load(args.feature_file)
    y = np.load(args.label_file)
    sampling_strategy = args.sampling_strategy
    assert sampling_strategy in ['ros', 'smote', 'rus', 'tomek', 'enn']
    if sampling_strategy == 'ros':
        assert args.ratio <= 1.0
        X_resampled, y_resampled = random_oversample(X, y, args.ratio)
    elif sampling_strategy == 'smote':
        assert args.ratio <= 1.0 
        X_resampled, y_resampled = SMOTE_oversample(X, y, args.ratio)
    elif sampling_strategy == 'rus':
        assert args.ratio <= 1.0
        X_resampled, y_resampled = random_undersample(X, y, args.ratio)
    elif sampling_strategy == 'tomek':
        X_resampled, y_resampled = tomek_undersample(X, y)
    else: 
        X_resampled, y_resampled = enn_undersample(X, y)
    print('Saving resampled feature vectors to {}'.format(args.output_dir))
    np.save(os.path.join(args.output_dir, args.output_feature_file), X_resampled)
    print('Saving resampled label vector to {}'.format(args.output_dir))
    np.save(os.path.join(args.output_dir, args.output_label_file), y_resampled)
