#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: RaulSierra
# @Date:   2014-09-11 11:25:37
# @Last Modified by:   RaulSierra
# @Last Modified time: 2014-09-16 10:04:05
import itertools as itools
import numpy as np
from sklearn import tree


class UnlikelyClassifier(object):

    '''
    classdocs
    '''

    def __init__(self, type="tree"):
        '''
        Constructor
        '''
        self.type = type
        self.models = dict()
        self.classes = None

    def fit(self, X, Y):
        self.classes = set(Y)
        nlevels = len(self.classes)

        for i in range(2, nlevels + 1):
            combs = itools.combinations(self.classes, i)
            ndata = len(X)
            for comb in combs:
                sub_X_indices = [i for i in range(ndata) if Y[i] in comb]

                print "Model for " + str(comb)
                print "With %i data" % len(sub_X_indices)
                model = tree.DecisionTreeClassifier()
                model.fit(X[sub_X_indices], Y[sub_X_indices])
                self.set_model(comb, model)

        print "Model dict keys: " + str(self.models.keys())

    def predict(self, X):
        predictions = self.__nested_predict(X, self.classes)

        return predictions

    def predict_proba(self, X):

        return NotImplementedError

    def __nested_predict(self, X, classes):
        model = self.get_model(classes)

        if len(classes) == 2:
            predictions = model.predict(X)
        else:
            class_probs = model.predict_proba(X)
            unlikely_classes_data = get_unlikely_classes(class_probs, model.classes_)
            unlikely_classes = set(unlikely_classes_data)
            predictions = np.zeros(len(X))

            # Para cada una de las clases improbables, dejamos 
            # solo los datos que tienen asignada esa clase como la menos probable y
            # los pasamos a un modelo que ya no considera esa clase  
            for clss in unlikely_classes:
                filter_idxs = np.where(unlikely_classes_data == clss)
                reduced_classes = [c for c in classes if c != clss]

                predictions[filter_idxs] = self.__nested_predict(
                    X[filter_idxs],
                    reduced_classes)

        return predictions

    def get_model(self, classes):
        model_key = ".".join([str(c) for c in sorted(classes)])
        return self.models[model_key]

    def set_model(self, classes, model):
        model_key = ".".join([str(c) for c in sorted(classes)])
        self.models[model_key] = model


def get_unlikely_classes(probs, classes):
    unlikely_classes = []
    for i, probs_v in enumerate(probs):
        class_i = np.argmin(probs_v)
        unlikely_classes.append(classes[class_i])

    return unlikely_classes
