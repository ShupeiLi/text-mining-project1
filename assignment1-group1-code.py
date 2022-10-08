# -*- coding: utf-8 -*-

"""
Group 1
Name and Student Number: 
Bangchao Xie, s3537145
Shupei Li, s3430863
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from itertools import product
import numpy as np
import math


seeds = 42


class NewsGroups():
    """
    Implement text categorization.
    Args:
        seeds: Set random state.
    """
    # The first parameter is the default value
    ## Task 4 parameters
    lowercase = list(product(["vect__lowercase"], [True, False]))
    stop_words = list(product(["vect__stop_words"], [None, "english"]))
    analyzer = list(product(list(product(["vect__analyzer"], ["word", "char", "char_wb"])), 
                            list(product(["vect__ngram_range"], [(1, 1), (1, 2), (2, 2)]))))
    max_features = list(product(["vect__max_features"], [None, 65000, 115000]))  # 0.5, 0.9 quantile

    # Classifier parameters
    nb_params = {"clf__alpha":[1, 0.1, 0.01, 0.0001]} # Naive Bayes
    svm_params = {"clf__alpha":[0.0001, 0.001, 0.01, 0.1]} # SVM
    rf_params = {"clf__n_estimators":[100, 50, 150, 200]} # Random Forest

    def __init__(self, seeds=1):
        self.train = fetch_20newsgroups(subset="train", shuffle=True, random_state=seeds)
        self.test = fetch_20newsgroups(subset="test", shuffle=True, random_state=seeds)
        self.file = open("./result.txt", "w")
        self._format()
        self._format(f"Data set\nCategories: {len(self.train.target_names)}\nTrain: {len(self.train.data)} records.\nTest: {len(self.test.data)} records.",
                     False)

    def _format(self, info="", line=True):
        if line:
            if info == "":
                msg = "=" * 60
            else:
                msg = "=" * math.floor((60 - len(info)) / 2) + info + "=" * math.ceil((60 - len(info)) / 2)
        else:
            msg = info
        print(msg)
        self.file.write(msg + "\n")

    def create_models(self):
        """
        Naive Bayes, SVM, Random Forest.
        """
        nb_count = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", MultinomialNB()),
        ])
        nb_tfidf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ])
        svm_count = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", SGDClassifier()),
        ])
        svm_tfidf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier()),
        ])
        rf_count = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", RandomForestClassifier()),
        ])
        rf_tfidf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestClassifier()),
        ])
        self.models = [nb_count, svm_count, rf_count, nb_tfidf, svm_tfidf, rf_tfidf]

    def run_models(self, params_dict=dict()):
        """
        Train and evaluate models.
        """
        def one_feature(num_lst, params=dict()):
            info = ["Naive Bayes", "SVM", "Random Forest"]
            classifiers = [NewsGroups.nb_params, NewsGroups.svm_params, NewsGroups.rf_params]
            for n in range(len(num_lst)):
                self._format(f"\n{info[n]}", False)
                i = num_lst[n]
                if params != dict():
                    self.models[i].set_params(**params)
                    self._format(f"params: {params}", False)

                # Parameter tuning
                self._format("Parameter tuning...", False)
                gs = GridSearchCV(self.models[i], classifiers[n], cv=5, n_jobs=-1)
                gs = gs.fit(self.train.data[:2000], self.train.target[:2000])
                self._format(f"Best parameters: {gs.best_params_}", False)
                self._format(f"Best score: {gs.best_score_:.6f}", False)

                # Train and evaluate the model
                self._format(f"Model training and evaluating...", False)
                self.models[i].set_params(**gs.best_params_)
                self.models[i].fit(self.train.data, self.train.target)
                predicted = self.models[i].predict(self.test.data)
                results = [np.mean(arr) for arr in metrics.precision_recall_fscore_support(self.test.target, predicted)]
                self._format(f"Precision: {results[0]:.6f}", False)
                self._format(f"Recall: {results[1]:.6f}", False)
                self._format(f"F1: {results[2]:.6f}", False)

        self.create_models()
        self._format("count")
        one_feature(list(range(3)), params_dict)
        self._format("tf")
        tf_params = {"tfidf__use_idf": False}
        tf_params.update(params_dict)
        one_feature(list(range(3, 6)), tf_params)
        self._format("tf-idf")
        one_feature(list(range(3, 6)), params_dict)
        self._format()

    def map_to_dict(self, one_param_lst):
        for item in one_param_lst:
            if type(item[0]) != tuple:
                yield {item[0]: item[1]}
            else:
                yield dict((key, value) for key, value in item)

    def one_epoch(self, task4=False, info="", task4_param=[]):
        self._format()
        self._format(info, False)
        if task4:
            task4 = self.map_to_dict(task4_param)
            for one_dict in task4:
                self.run_models(one_dict)
        else:
            self.run_models()

    def main(self):
        """
        Experiments.
        """
        # Task 2 and Task 3
        self.one_epoch(False, "Task 2 and 3: Default Parameters")

        # Task 4
        ## a. lowercase
        self.one_epoch(True, "Task 4.a: lowercase", NewsGroups.lowercase)

        ## b. stop_words
        self.one_epoch(True, "Task 4.b: stop_words", NewsGroups.stop_words)
        
        ## c. analyzer
        self.one_epoch(True, "Task 4.c: analyzer", NewsGroups.analyzer)
        
        ## d. max_features
        self.one_epoch(True, "Task 4.d: max_features", NewsGroups.max_features)

        self.file.close()

if __name__ == "__main__":
    model = NewsGroups(seeds)
    model.main()
