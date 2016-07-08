from __future__ import print_function, division

# Get source directory
import sys
import os

root = os.getcwd().split('src')[0] + 'src'
sys.path.append(os.path.abspath(root))
import click
import numpy as np
from sklearn import svm
from injest import Vessel
from TermFrequency import TermFrequency
from time import time
from random import shuffle
from ES_CORE import ESHandler
from ABCD import ABCD
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from funcs import csr_dot
from pdb import set_trace

__author__ = 'Rahul Krishna'

OPT = Vessel(
        FORCE_INJEST=False,
        VERBOSE_MODE=False)


def masking(csrmat, mask):
    if mask == []:
        return csrmat
    tmp = np.identity(csrmat.shape[1])
    for x in mask:
        tmp[x, x] = 0;
    my_mask = csr_matrix(tmp)
    re=csrmat * my_mask
    return re


class FeatureMap:
    def __init__(self, raw_data, n=1000, scroll=None, features=None):
        self.data = raw_data
        self._class = list()
        self.doc_id = list()
        self.mappings = list()
        self.features = features if features is not None else range(
                len(self.data['header']))
        self.scroll = scroll
        self._init_mapping()

    def _init_mapping(self):
        vocab = self.data['header']
        self.header = [vocab[feat] for feat in self.features]
        numel = range(len(self.data["meta"]))
        if self.scroll:
            shuffle(numel)
        self.mappings = self.data["matrix"][numel[:self.scroll]]
        for idx in numel[:self.scroll]:
            # instance = self.data["matrix"][idx]
            # self.mappings = csr_vstack(self.mappings, instance)
            self._class.append(self.data["meta"][idx]['label'])
            self.doc_id.append(self.data["meta"][idx]['doc_id'])
        ### masking ###
        self.mappings = self.mappings[:, self.features]




    # def refresh_label(self):
    #     self._class=[ESHandler().get_document(_id)["_source"]["label"] for _id in self.doc_id]

    def tf_idf(self, mask=[]):
        self._ifeatures = masking(self.mappings[:self.scroll], mask)
        transformer = TfidfTransformer(norm='l2', use_idf=True,
                                       smooth_idf=True, sublinear_tf=True)
        self._ifeatures = transformer.fit_transform(self._ifeatures,
                                                    self._class)
        return self

    def tf(self, mask=[]):
        self._ifeatures = masking(self.mappings, mask)
        transformer = TfidfTransformer(norm='l2', use_idf=False,
                                       smooth_idf=True, sublinear_tf=False)
        self._ifeatures = transformer.fit_transform(self._ifeatures)
        return self


class SVM:
    "Classifiers"

    def __init__(self, disp, opt=None):

        if opt:
            global OPT
            OPT = opt

        self.disp = disp
        self.TF = TermFrequency(site='english',
                                force_injest=OPT.FORCE_INJEST,
                                verbose=OPT.VERBOSE_MODE)
        self.helper = ESHandler(es=self.TF.es, force_injest=False)

        # Initialize attributes
        self.round = 0
        self.result = list()
        self.CONTROL = None

    @staticmethod
    def vprint(string):
        if OPT.VERBOSE_MODE:
            print(string)

    @staticmethod
    def fselect(all_docs, n_features=4000):
        transformer = TfidfTransformer(norm='l2', use_idf=True
                                       , smooth_idf=True, sublinear_tf=True)
        tfidf_mtx = transformer.fit_transform(all_docs)
        key_features = np.argsort(tfidf_mtx.sum(axis=0)).tolist()[0][
                       -n_features:]
        return key_features

    def featurize(self):
        t = time()
        train_tfm = self.TF.matrix(CONTROL=False, LABELED=True)
        all_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        self.vprint("Get TFM. {} seconds elapsed".format(time() - t))
        t = time()
        self.top_feat = self.fselect(all_docs=all_tfm["matrix"])
        self.vprint("Feature selection. {} seconds elapsed".format(time() - t))
        t = time()

        ## Save TF-IDF score
        self.vocab = [train_tfm['header'][i] for i in self.top_feat]
        # self.TRAIN = FeatureMap(raw_data=train_tfm,features=self.top_feat).tf()
        self.vprint("Featurization. {} seconds elapsed".format(time() - t))
        return self

    def update_matrix(self):
        pass

    def rerun(self, mask=list()):
        "Masking"
        tmp = np.identity(len(self.vocab))
        for x in mask:
            tmp[x, x] = 0;
        my_mask = csr_matrix(tmp)
        TO_REVIEW = [self.TEST.pop(idx) for idx in
                     self.sort_order_uncertain[:50]]
        self.TRAIN.update_mapping(TO_REVIEW)
        return self

    def run(self, mask=[]):
        self.round += 1
        self.clf = svm.SVC(kernel='linear', probability=True)

        # Update training matrix
        t = time()
        train_tfm = self.TF.matrix(CONTROL=False, LABELED=True)
        # self.TRAIN = self.TRAIN.refresh(data=train_tfm)
        self.TRAIN = FeatureMap(raw_data=train_tfm,
                   features=self.top_feat).tf(mask=mask)

        self.vprint("UPDATE TFM for TRAIN. {} seconds elapsed".format(time() - t))
        t = time()

        self.clf.fit(self.TRAIN._ifeatures, self.TRAIN._class)

        self.vprint("TRAIN SVM. {} seconds elapsed".format(time() - t))
        t = time()

        # Update test matrix
        test_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        self.vprint("Load from ES for TEST. {} seconds elapsed".format(time() - t))
        t = time()
        self.TEST = FeatureMap(raw_data=test_tfm,
                               features=self.top_feat).tf(mask=mask)

        self.vprint("Get TFM for TEST. {} seconds elapsed".format(time() - t))
        t = time()

        pred_proba = self.clf.predict_proba(self.TEST._ifeatures)
        pos_at = list(self.clf.classes_).index("pos")
        self.coef = self.clf.coef_.toarray()[0]
        self.dual_coef = self.clf.dual_coef_.toarray()[0]

        if not pos_at:
            self.coef = -self.coef
            self.dual_coef = -self.dual_coef

        support = self.clf.support_
        self.prob = pred_proba[:, pos_at]
        self.sort_order_uncertain = np.argsort(np.abs(pred_proba[:, 0] - 0.5))
        self.sort_order_certain = np.argsort(1 - self.prob)
        self.sort_order_support = np.argsort(1 - np.abs(self.dual_coef))

        self.certain = [self.helper.get_document(
                _id=self.TEST.doc_id[i]) for i in
                        self.sort_order_certain[:self.disp]]
        self.uncertain = [self.helper.get_document(
                _id=self.TEST.doc_id[i]) for i in
                          self.sort_order_uncertain[:self.disp]]
        self.support_vec = [self.helper.get_document(
                _id=self.TRAIN.doc_id[i]) for i in
                            support[self.sort_order_support[:self.disp]]]

        self.vprint("SUMMARIZED. {} seconds elapsed".format(time() - t))
        t = time()

        return self.stats()

    def stats(self):
        t = time()

        # --Stats--

        self.CONTROL = FeatureMap(raw_data=self.TF.matrix(CONTROL=True, LABELED=True), features=self.top_feat).tf()

        self.vprint("Get TFM for CONTROL. {} seconds elapsed".format(time() - t))
        t = time()


        # Turnovers
        preds = self.clf.predict(self.CONTROL._ifeatures)
        pred_proba = self.clf.predict_proba(self.CONTROL._ifeatures)
        pos_at = list(self.clf.classes_).index("pos")
        self.proba=pred_proba[:, pos_at]
        turnover = [i for i in xrange(len(self.proba)) if not self.CONTROL._class[i]==preds[i]]
        sort_order = np.argsort(0.5-np.abs(self.proba[turnover]-0.5))
        self.real_order = np.array(turnover)[sort_order][:self.disp]
        self.turnovers = [self.helper.get_document(_id=self.CONTROL.doc_id[i]) for i in self.real_order]

        self.vprint("TURNOVERS. {} seconds elapsed".format(time() - t))
        t = time()
        #######
        # Determinants (Nearest neighbors to turnovers in training set)
        self.bests=[]
        self.best_dists=[]
        for i in self.real_order:
            for j, can in enumerate(self.TRAIN._ifeatures):
                if not j:
                    best=j
                    best_dist=csr_dot(self.CONTROL._ifeatures[i],can)
                else:
                    dist=csr_dot(self.CONTROL._ifeatures[i], can)
                    if dist>best_dist:
                        best_dist=dist
                        best=j
            self.bests.append(best)
            self.best_dists.append(best_dist)
        self.determinants = [self.helper.get_document(_id=self.TRAIN.doc_id[i]) for i in self.bests]
        self.vprint("DETERMINENTS. {} seconds elapsed".format(time() - t))
        t = time()

        #######

        # Stats
        self.abcd = ABCD(before=self.CONTROL._class, after=preds)
        self.STATS = [k.stats() for k in self.abcd()]

        self.vprint("Get STATS. {} seconds elapsed".format(time() - t))
        t = time()
        ###########

        self.result.append({
            "the_round" : self.round,
            "consistency": 0,
            "turnover_prob": ','.join(
                    map(str, self.proba[self.real_order])),
            "turnover"  : self.turnovers,
            "pos"        : self.STATS[1],
            "neg"        : self.STATS[0]
        })
        return self

    def get_response(self):
        return {
            "the_round"     : self.round,
            "coef"          : ','.join(map(str, self.coef)),
            "support"       : self.support_vec,
            "dual_coef"     : ','.join(
                    map(str, self.dual_coef[self.sort_order_support])),
            "certain_prob"  : ','.join(
                map(str, self.prob[self.sort_order_certain])),
            "uncertain_prob": ','.join(
                    map(str, self.prob[self.sort_order_uncertain])),
            "certain"       : self.certain,
            "uncertain"     : self.uncertain[::-1],
            "vocab"         : self.vocab,
            "the_round": self.round,
            "consistency"   : 0,
            "turnover_prob" : ','.join(
                map(str, self.proba[self.real_order])),
            "turnover"      : self.turnovers,
            "pos"           : self.STATS[1],
            "neg"           : self.STATS[0],
            "determinant"   : self.determinants,
            "determinant_dist": ','.join(map(str, self.best_dists))
        }


# ----Command Line Interface----
@click.command()
@click.option('--force', default="false",
              help='Flags: True/False. Create force injest documents to ES.')
@click.option('--debug', default="false",
              help='Flags: True/False. Enter verbose mode (display all '
                   'outputs to stdout.\n')
def cmd(force, debug):
    global OPT
    OPT.also(
            FORCE_INJEST=force.lower() == 'true',
            VERBOSE_MODE=debug.lower() == 'true'
    )
    # set_trace()
    print(
            "\nRunning: model.py with settings: \n\t--force={FORCE}\n "
            "\t--debug={DEBUG}".format(
                    FORCE=force,
                    DEBUG=debug))
    classifier = SVM()
    return classifier.run()


if __name__ == '__main__':
    cmd()
