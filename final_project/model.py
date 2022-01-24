import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from feature_selector import FeatureSelector
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

from preprocessing import data_preprocessing

def AdaBoost(x_train, x_valid, y_train, y_valid, train_df, test_df):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME",n_estimators=200, learning_rate=0.8)
    bdt.fit(x_train, y_train)
    ans = pd.read_csv(r"./data/submission.csv", encoding="utf-8")
    ans["LEVEL"] = bdt.predict(test_df).astype(int)
    ans.to_csv("adaboost.csv", encoding="utf-8", index=False)


def RandomForest(x_train, x_valid, y_train, y_valid, train_df, test_df):
    clf = RandomForestClassifier()
    p = {
        "n_estimators":range(50, 80),
        "max_depth":range(10,20)
    }
    clf = RandomForestClassifier(n_estimators=73, max_depth=12)
    clf.fit(x_train, y_train)
    ans = pd.read_csv(r"./data/submission.csv", encoding="utf-8")
    ans["LEVEL"] = clf.predict(test_df).astype(int)
    ans.to_csv("randomforest.csv", encoding="utf-8", index=False)


def GradientBoosting(x_train, x_valid, y_train, y_valid, train_df, test_df):
    model = GradientBoostingClassifier(n_estimators=300)
    model.fit(x_train,y_train)
    ans = pd.read_csv(r"./data/submission.csv", encoding="utf-8")
    ans["LEVEL"] = model.predict(test_df).astype(int)
    ans.to_csv("GBC.csv", encoding="utf-8", index=False)


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid, train_df, test_df = data_preprocessing()
    AdaBoost(x_train, x_valid, y_train, y_valid, train_df, test_df)
    RandomForest(x_train, x_valid, y_train, y_valid, train_df, test_df)
    GradientBoosting(x_train, x_valid, y_train, y_valid, train_df, test_df)