import pandas as pd
import csv
import numpy as np
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from feature_selector import FeatureSelector
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def data_preprocessing():

    train_df = pd.read_csv("train.csv", encoding="utf-8")
    test_df = pd.read_csv(r"test.csv", encoding="utf-8")

    # add '0' to NaN at train_df
    train_df["1暴露岩岸"] = train_df["1暴露岩岸"].fillna('0')
    train_df["2暴露人造結構物"] = train_df["2暴露人造結構物"].fillna('0')
    train_df["3暴露岩盤"] = train_df["3暴露岩盤"].fillna('0')
    train_df["4沙灘"] = train_df["4沙灘"].fillna('0')
    train_df["5砂礫混合灘"] = train_df["5砂礫混合灘"].fillna('0')
    train_df["6礫石灘"] = train_df["6礫石灘"].fillna('0')
    train_df["7開闊潮間帶"] = train_df["7開闊潮間帶"].fillna('0')
    train_df["8遮蔽岩岸"] = train_df["8遮蔽岩岸"].fillna('0')
    train_df["9遮蔽潮間帶"] = train_df["9遮蔽潮間帶"].fillna('0')
    train_df["10遮蔽濕地"] = train_df["10遮蔽濕地"].fillna('0')


    # add '0' to NaN at test_df
    test_df["1暴露岩岸"] = test_df["1暴露岩岸"].fillna('0')
    test_df["2暴露人造結構物"] = test_df["2暴露人造結構物"].fillna('0')
    test_df["3暴露岩盤"] = test_df["3暴露岩盤"].fillna('0')
    test_df["4沙灘"] = test_df["4沙灘"].fillna('0')
    test_df["5砂礫混合灘"] = test_df["5砂礫混合灘"].fillna('0')
    test_df["6礫石灘"] = test_df["6礫石灘"].fillna('0')
    test_df["7開闊潮間帶"] = test_df["7開闊潮間帶"].fillna('0')
    test_df["8遮蔽岩岸"] = test_df["8遮蔽岩岸"].fillna('0')
    test_df["9遮蔽潮間帶"] = test_df["9遮蔽潮間帶"].fillna('0')
    test_df["10遮蔽濕地"] = test_df["10遮蔽濕地"].fillna('0')

    train_df = train_df.drop(["Location", "County", "Station"], axis=1)
    test_df = test_df.drop(["Location", "County", "Station"], axis=1)

    y_train = train_df['LEVEL']
    x_train = train_df.drop(['LEVEL'], axis=1)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

    return x_train, x_valid, y_train, y_valid, train_df, test_df

