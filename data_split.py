from os import name
import numpy as np 
from sklearn.model_selection import train_test_split
import argparse
import os 
def load_data(feature_file, label_file):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', type=str)
    parser.add_argument('--label_file', type=str)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--do_split_val', action='store_true', help="Whether to split train data into train and dev set")
    parser.add_argument('--val_size', default=0.2, type=float)
    parser.add_argument('--output_dir', default='data', type=str)
    args = parser.parse_args()

    X, y = load_data(args.feature_file, args.label_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)
    if args.do_split_val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val_size, random_state=0)
        print('Saving train feature vectors to {}'.format(os.path.join(args.output_dir, 'train_features.npy')))
        np.save(os.path.join(args.output_dir, 'train_features.npy'), X_train)
        print('Saving train label vector to {}'.format(os.path.join(args.output_dir, 'train_labels.npy')))
        np.save(os.path.join(args.output_dir, 'train_labels.npy'), y_train)
        print('Saving validation feature vectors to {}'.format(os.path.join(args.output_dir, 'val_features.npy')))
        np.save(os.path.join(args.output_dir, 'val_features.npy'), X_val)
        print('Saving validation label vector to {}'.format(os.path.join(args.output_dir, 'val_labels.npy')))
        np.save(os.path.join(args.output_dir, 'val_labels.npy'), y_val)
        print('Saving test feature vectors to {}'.format(os.path.join(args.output_dir, 'test_features.npy')))
        np.save(os.path.join(args.output_dir, 'test_features.npy'), X_test)
        print('Saving test label vector to {}'.format(os.path.join(args.output_dir, 'test_labels.npy')))
        np.save(os.path.join(args.output_dir, 'test_labels.npy'), y_test)
    else: 
        print('Saving train feature vectors to {}'.format(os.path.join(args.output_dir, 'train_features.npy')))
        np.save(os.path.join(args.output_dir, 'train_features.npy'), X_train)
        print('Saving train label vector to {}'.format(os.path.join(args.output_dir, 'train_labels.npy')))
        np.save(os.path.join(args.output_dir, 'train_labels.npy'), y_train)
        print('Saving test feature vectors to {}'.format(os.path.join(args.output_dir, 'test_features.npy')))
        np.save(os.path.join(args.output_dir, 'test_features.npy'), X_test)
        print('Saving test label vector to {}'.format(os.path.join(args.output_dir, 'test_labels.npy')))
        np.save(os.path.join(args.output_dir, 'test_labels.npy'), y_test)

