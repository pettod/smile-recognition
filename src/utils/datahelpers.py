"""datahelpers.py

Helper functions for handling data
"""
import os
import glob

import cv2 as cv
from sklearn.model_selection import train_test_split


def load_labels(root: str):
    """Load labeld from the path"""
    path = os.path.join(root, 'labels.txt')
    with open(path, 'r') as gt_file:
        labels = [float(line.rstrip().split()[0]) for line in gt_file]
    return labels


def load_imgs(root: str):
    """Load image names from the root directory"""
    path = os.path.join(root, 'files')
    imgs = [cv.resize(cv.imread(os.path.abspath(img)), (64, 64)) for img in glob.glob(path + '/*jpg')]
    return imgs


def split_data(imgs: list, labels: list):
    """Make a 80-20 split of the data using stratified randomness"""
    assert len(imgs) == len(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        imgs, labels, test_size=0.2, random_state=42, stratify=labels
        )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # This is just a test that the data fits the precondition that the number
    # of images equal the number of labels, and print the names and labels of
    # the test files
    data_loc = 'data/genki4k/'
    ims = [i for i in glob.glob(os.path.join(data_loc, 'files') + '/*jpg')]
    # ims = load_imgs(data_loc)
    lbl = load_labels(data_loc)
    print(len(lbl) == len(ims))
    _, x_tst, _, y_tst = split_data(ims, lbl)
    print(x_tst, y_tst)
