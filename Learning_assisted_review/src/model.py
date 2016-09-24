from __future__ import print_function, division

# Get source directory
import sys
import os

root = os.getcwd().split('src')[0] + 'src'
sys.path.append(os.path.abspath(root))
import click
from collections import Counter
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
from funcs import *
from pdb import set_trace
import pickle

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
        self.user=list()
        self.doc_id = list()
        self.mappings = list()
        self.year=list()
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
            self.user.append(self.data["meta"][idx]['user'])
            self.doc_id.append(self.data["meta"][idx]['doc_id'])
            self.year.append(self.data["meta"][idx]['year'])
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

    def __init__(self, disp, set = 'Hall', opt=None):

        if opt:
            global OPT
            OPT = opt

        self.set = set
        self.disp = disp
        self.TF = TermFrequency(site=self.set,
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



####################################################

    def simple_active(self, step=10 ,initial=200, pos_limit=5, margin=1, mask=[]):
        all_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        collection = FeatureMap(raw_data=all_tfm,
                                features=self.top_feat).tf(mask=mask)
        csr_mat=collection._ifeatures
        labels=np.array(collection.user)
        num=len(labels)
        pool=range(num)
        train=[]
        steps = np.array(range(int(num / step))) * step

        pos=0
        pos_track=[0]
        is_stable=False
        clf = svm.SVC(kernel='linear', probability=True)
        start=0
        stable=0
        begin=0
        result={}
        for idx, round in enumerate(steps[:-1]):
            can = np.random.choice(pool, step, replace=False)
            train.extend(can)
            pool = list(set(pool) - set(can))
            try:
                pos = Counter(labels[train])["yes"]
            except:
                pos = 0
            pos_track.append(pos)

            if not begin:
                pool2=pool[:]
                train2=train[:]
                pos_track2=pos_track[:]
                pool3 = pool2[:]
                train3 = train2[:]
                pos_track3 = pos_track2[:]
                pool4 = pool2[:]
                train4 = train2[:]
                pos_track4 = pos_track2[:]
                pool7 = pool2[:]
                train7 = train2[:]
                pos_track7 = pos_track2[:]
                if round >= initial and pos>=pos_limit:
                    begin=idx+1
            else:
                clf.fit(csr_mat[train4], labels[train4])
                pred_proba4 = clf.predict_proba(csr_mat[pool4])
                pos_at = list(clf.classes_).index("yes")
                proba4 = pred_proba4[:, pos_at]
                sort_order_certain4 = np.argsort(1 - proba4)
                can4 = [pool4[i] for i in sort_order_certain4[:step]]
                train4.extend(can4)
                pool4 = list(set(pool4) - set(can4))
                pos = Counter(labels[train4])["yes"]
                pos_track4.append(pos)

                ### continuous aggressive
                clf.fit(csr_mat[train7], labels[train7])
                poses = np.where(labels[train7] == "yes")[0]
                negs = np.where(labels[train7] == "no")[0]
                train_dist = clf.decision_function(csr_mat[train7][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample7 = np.array(train7)[poses].tolist() + np.array(train7)[negs][negs_sel].tolist()

                clf.fit(csr_mat[sample7], labels[sample7])
                pred_proba7 = clf.predict_proba(csr_mat[pool7])
                pos_at = list(clf.classes_).index("yes")
                proba7 = pred_proba7[:, pos_at]
                sort_order_certain7 = np.argsort(1 - proba7)
                can7 = [pool7[i] for i in sort_order_certain7[:step]]
                train7.extend(can7)
                pool7 = list(set(pool7) - set(can7))
                pos = Counter(labels[train7])["yes"]
                pos_track7.append(pos)


                if not is_stable:
                    clf.fit(csr_mat[train2], labels[train2])
                    pred_proba = clf.predict_proba(csr_mat[pool2])
                    # sort_order_uncertain = np.argsort(np.abs(pred_proba[:,0] - 0.5))
                    dist = clf.decision_function(csr_mat[pool2])
                    sort_order_dist = np.argsort(np.abs(dist))
                    if abs(dist[sort_order_dist[0]]) > margin or round == steps[-2]:
                        is_stable = True
                        stable=idx


                        train5 = train2[:]
                        pos_track5 = pos_track2[:]
                        train6 = train2[:]
                        pos_track6 = pos_track2[:]


                        pos_at = list(clf.classes_).index("yes")
                        proba = pred_proba[:, pos_at]
                        sort_order_certain2 = np.argsort(1 - proba)
                        can2 = [pool2[i] for i in sort_order_certain2[start:start + step]]

                        train2.extend(can2)
                        pos = Counter(labels[train2])["yes"]
                        pos_track2.append(pos)


                        ### data balancing ###
                        ### Agressive undersampling ####
                        poses=np.where(labels[train5] == "yes")[0]
                        negs=np.where(labels[train5] == "no")[0]
                        train_dist = clf.decision_function(csr_mat[train5][negs])
                        negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                        sample5 = np.array(train5)[poses].tolist()+np.array(train5)[negs][negs_sel].tolist()
                        clf.fit(csr_mat[sample5], labels[sample5])
                        pred_proba5 = clf.predict_proba(csr_mat[pool2])
                        pos_at5 = list(clf.classes_).index("yes")
                        proba5 = pred_proba5[:, pos_at5]
                        sort_order_certain5 = np.argsort(1 - proba5)
                        can5 = [pool2[i] for i in sort_order_certain5[start:start + step]]
                        train5.extend(can5)
                        pos = Counter(labels[train5])["yes"]
                        pos_track5.append(pos)

                        ### SMOTE ####
                        # negs_sel = np.argsort(np.abs(train_dist))[::-1][:int(0.5*len(train6))]
                        # sample6 = np.array(train6)[poses].tolist() + np.array(train6)[negs][negs_sel].tolist()
                        # csr_train6, label_train6 = smote_most(csr_mat[sample6], labels[sample6])
                        # clf.fit(csr_train6, label_train6)
                        # pred_proba6 = clf.predict_proba(csr_mat[pool2])
                        # pos_at6 = list(clf.classes_).index("yes")
                        # proba6 = pred_proba6[:, pos_at6]
                        # sort_order_certain6 = np.argsort(1 - proba6)
                        # can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                        # train6.extend(can6)
                        # pos = Counter(labels[train6])["yes"]
                        # pos_track6.append(pos)

                        #####################

                        pool3 = list(set(pool2) - set(can5))
                        train3 = train5[:]
                        pos_track3 = pos_track5[:]

                        pool8 = pool3[:]
                        train8 = train3[:]
                        pos_track8 = pos_track3[:]

                        start = start + step
                    else:
                        # can2 = [pool2[i] for i in sort_order_uncertain[:step]]
                        can2 = [pool2[i] for i in sort_order_dist[:step]]
                        train2.extend(can2)
                        pool2 = list(set(pool2) - set(can2))
                        pos = Counter(labels[train2])["yes"]
                        pos_track2.append(pos)

                else:
                    #### semi_continuous_aggressive
                    clf.fit(csr_mat[train3], labels[train3])
                    poses = np.where(labels[train3] == "yes")[0]
                    negs = np.where(labels[train3] == "no")[0]
                    train_dist = clf.decision_function(csr_mat[train3][negs])
                    negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                    sample3 = np.array(train3)[poses].tolist() + np.array(train3)[negs][negs_sel].tolist()

                    clf.fit(csr_mat[sample3], labels[sample3])
                    pred_proba3 = clf.predict_proba(csr_mat[pool3])
                    pos_at = list(clf.classes_).index("yes")
                    proba3 = pred_proba3[:, pos_at]
                    sort_order_certain3 = np.argsort(1 - proba3)
                    can3 = [pool3[i] for i in sort_order_certain3[:step]]
                    train3.extend(can3)
                    pool3 = list(set(pool3) - set(can3))
                    pos = Counter(labels[train3])["yes"]
                    pos_track3.append(pos)

                    #### semi_continuous
                    clf.fit(csr_mat[train8], labels[train8])
                    pred_proba8 = clf.predict_proba(csr_mat[pool8])
                    pos_at = list(clf.classes_).index("yes")
                    proba8=pred_proba8[:,pos_at]
                    sort_order_certain8 = np.argsort(1-proba8)
                    can8 = [pool8[i] for i in sort_order_certain8[:step]]
                    train8.extend(can8)
                    pool8 = list(set(pool8) - set(can8))
                    pos = Counter(labels[train8])["yes"]
                    pos_track8.append(pos)

                    #################################

                    can2 = [pool2[i] for i in sort_order_certain2[start:start + step]]
                    train2.extend(can2)
                    pos = Counter(labels[train2])["yes"]
                    pos_track2.append(pos)


                    can5 = [pool2[i] for i in sort_order_certain5[start:start + step]]
                    train5.extend(can5)
                    pos = Counter(labels[train5])["yes"]
                    pos_track5.append(pos)

                    # can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                    # train6.extend(can6)
                    # pos = Counter(labels[train6])["yes"]
                    # pos_track6.append(pos)

                    start = start + step


            print("Round #{id} passed\r".format(id=round), end="")

        result["begin"] = begin
        result["stable"] = stable
        result["x"] = steps
        result["linear_review"] = pos_track
        result["simple_active"] = pos_track2
        result["semi_continuous_aggressive"] = pos_track3
        result["continuous_active"] = pos_track4
        result["aggressive_undersampling"] = pos_track5
        # result["smote"] = pos_track6
        result["continuous_aggressive"] = pos_track7
        result["semi_contunuous"] = pos_track8

        return result

    def splitData(self,year):
        all_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        collection = FeatureMap(raw_data=all_tfm,
                                features=self.top_feat).tf()

        range1=[]
        range2=[]
        for i in xrange(len(collection.year)):
            try:
                if int(collection.year[i])<=int(year):
                    range1.append(i)
                else:
                    range2.append(i)
            except:
                pass


        csr1=collection._ifeatures[range1]
        labels1=np.array(collection.user)[range1]
        csr2=collection._ifeatures[range2]
        labels2=np.array(collection.user)[range2]
        with open("../dump/"+self.set+str(year)+".pickle","wb") as handle:
            pickle.dump(csr1, handle)
            pickle.dump(labels1, handle)
        with open("../dump/"+self.set+str(2010)+".pickle","wb") as handle:
            pickle.dump(csr2, handle)
            pickle.dump(labels2, handle)

    def saveData(self):
        all_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        collection = FeatureMap(raw_data=all_tfm,
                                features=self.top_feat).tf()
        csr_mat=collection._ifeatures
        labels=np.array(collection.user)
        with open("../dump/"+self.set+".pickle","wb") as handle:
            pickle.dump(csr_mat, handle)
            pickle.dump(labels, handle)
        with open("../dump/"+self.set+".pickle","rb") as f:
            csr = pickle.load(f)
            lb = pickle.load(f)
            set_trace()



    def linear_review(self, step=10, mask=[]):
        all_tfm = self.TF.matrix(CONTROL=False, LABELED=False)
        collection = FeatureMap(raw_data=all_tfm,
                                features=self.top_feat).tf(mask=mask)
        csr_mat = collection._ifeatures
        labels = np.array(collection.user)
        num = len(labels)
        pool = range(num)
        train = []
        steps = np.array(range(int(num / step))) * step
        pos = 0
        pos_track = [0]
        for round in steps[:-1]:
            can = np.random.choice(pool, step, replace=False)

            train.extend(can)
            pool = list(set(pool) - set(can))
            try:
                pos = Counter(labels[train])["yes"]
            except:
                pos = 0
            pos_track.append(pos)
            print("Round #{id} passed\r".format(id=round), end="")
        return steps, pos_track







######################################################




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
