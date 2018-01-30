from sklearn import svm
from sklearn import linear_model
import numpy as np
from pdb import set_trace

class Esvm(object):
    def __init__(self,level = 3):
        self.level = level
        self.model = None
        self.svms = [svm.SVC(kernel='linear', probability=True, class_weight={'yes':2**i,'no':1}) for i in xrange(-level, level+1)]
        self.lgr = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
        self.classes_=['yes','no']

    def fit(self, body, label):
        scores = []
        labels = [1 if l=='yes' else 0 for l in label]
        for clf in self.svms:
            clf.fit(body,label)
            pos_at = list(clf.classes_).index("yes")
            scores.append(clf.predict_proba(body)[:,pos_at])


        self.lgr.fit(np.transpose(np.array(scores)),labels)


    def decision_function(self, body):
        scores = []
        for clf in self.svms:
            pos_at = list(clf.classes_).index("yes")
            scores.append(clf.predict_proba(body)[:, pos_at])
        return self.lgr.decision_function(np.transpose(np.array(scores)))

    def predict(self, body):
        scores = []
        for clf in self.svms:
            pos_at = list(clf.classes_).index("yes")
            scores.append(clf.predict_proba(body)[:, pos_at])
        poses = list(self.lgr.classes_).index(1)
        df = self.lgr.predict_proba(np.transpose(np.array(scores)))[:,poses]
        return np.array(['yes' if x>=0.5 else 'no' for x in df])

    def predict_proba(self, body):

        scores = []
        for clf in self.svms:
            pos_at = list(clf.classes_).index("yes")
            scores.append(clf.predict_proba(body)[:, pos_at])
        poses = list(self.lgr.classes_).index(1)
        df = self.lgr.predict_proba(np.transpose(np.array(scores)))[:,poses]

        return np.transpose(np.array([df,1-df]))

