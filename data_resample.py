from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import argparse

def random_oversample(X_train, y_train, ratio=1):
    ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    return ros.fit_sample(X_train, y_train)
def SMOTE_oversample(X_train, y_train, ratio=1):
    smote = SMOTE(sampling_strategy=ratio)
    return smote.fit_sample(X_train, y_train)
def random_undersample(X_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    return rus.fit_sample(X_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    X_train_ros_11, y_train_ros_11 = random_oversample(X_train_transformed, y_train, ratio=1)
    X_train_ros_23, y_train_ros_23 = random_oversample(X_train_transformed, y_train, ratio=2/3)
