from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os

class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.kept=50
        self.atleast=100
        self.syn_thres = 0.9


    def create(self,filename):
        self.filename=filename
        self.name=self.filename.split(".")[0]
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[]}
        self.body={}
        self.est=[]
        self.last_pos=0
        self.last_neg=0
        self.record_est={"x":[],"semi":[],"sigmoid":[]}


        try:
            ## if model already exists, load it ##
            return self.load()
        except:
            ## otherwise read from file ##
            try:
                self.loadfile()
                self.preprocess()
                self.save()
            except:
                ## cannot find file in workspace ##
                self.flag=False
        return self

    ### Use previous knowledge, labeled only
    def create_old(self, filename):
        with open("../workspace/coded/" + str(filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "code", "time"]
        header = content[0]
        ind0 = header.index("code")
        self.last_pos = len([c[ind0] for c in content[1:] if c[ind0] == "yes"])
        self.last_neg = len([c[ind0] for c in content[1:] if c[ind0] == "no"])
        for field in fields:
            ind = header.index(field)
            if field == "time":
                self.body[field].extend([float(c[ind]) for c in content[1:] if c[ind0] != "undetermined"])
            else:
                self.body[field].extend([c[ind] for c in content[1:] if c[ind0] != "undetermined"])
        try:
            ind = header.index("label")
            self.body["label"].extend([c[ind] for c in content[1:] if c[ind0]!="undetermined"])
        except:
            self.body["label"].extend([c[ind0] for c in content[1:] if c[ind0]!="undetermined"])

        self.preprocess()
        self.save()


    def loadfile(self):
        with open("../workspace/data/" + str(self.filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        header = content[0]
        for field in fields:
            ind = header.index(field)
            self.body[field] = [c[ind] for c in content[1:]]
        try:
            ind = header.index("label")
            self.body["label"] = [c[ind] for c in content[1:]]
        except:
            self.hasLabel=False
            self.body["label"] = ["unknown"] * (len(content) - 1)
        try:
            ind = header.index("code")
            self.body["code"] = [c[ind] for c in content[1:]]
        except:
            self.body["code"]=['undetermined']*(len(content) - 1)
        try:
            ind = header.index("time")
            self.body["time"] = [c[ind] for c in content[1:]]
        except:
            self.body["time"]=[0]*(len(content) - 1)
        return
    
    def lda(self):
        import lda
        from scipy.sparse import csr_matrix

        def normalize(x, p=2):
            xx = np.linalg.norm(x, p)
            return x / xx if xx else x

        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
                               vocabulary=self.voc, decode_error="ignore")
        self.csr_mat = tfer.fit_transform(content)

        lda1 = lda.LDA(n_topics=100, alpha=0.1, eta=0.01, n_iter=200)
        self.csr_mat = csr_matrix([normalize(row) for row in lda1.fit_transform(self.csr_mat.astype(int))])
        return

    def doc2vec(self):
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        import multiprocessing

        def convert_sentences(sentence_list):
            for i in range(len(sentence_list)):
                for char in ['.', ',', '!', '?', ';', ':']:
                    sentence_list[i] = sentence_list[i].replace(char, ' ' + char + ' ')
            return [TaggedDocument(words=sentence_list[i].split(), tags=[i]) for i in range(len(sentence_list))]

        def normalize(x, p=2):
            xx = np.linalg.norm(x, p)
            return x / xx if xx else x

        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]

        content1 = convert_sentences(content)
        model = Doc2Vec(size=300, window=10, min_count=5, workers=multiprocessing.cpu_count(),alpha=0.025, min_alpha=0.025)
        model.build_vocab(content1)

        for epoch in range(10):
            model.train(content1, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        self.csr_mat = np.array([normalize(model.infer_vector(x.words, alpha=model.alpha, min_alpha=model.min_alpha),p=2) for x in content1])
        return
    
    def syn_error(self):
        tmp = [Counter([self.body['label'][j] for j, vecb in enumerate(self.csr_mat) if
                                           (vec * vecb.transpose()).toarray()[0, 0] >= self.syn_thres]) for vec in
                                  self.csr_mat]
        self.body['syn_error'] = [x['yes']/sum(x.values()) for x in tmp]
        set_trace()

    def export_feature(self):
        with open("../workspace/coded/feature_" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for i in xrange(self.csr_mat.shape[0]):
                for j in range(self.csr_mat.indptr[i],self.csr_mat.indptr[i+1]):
                    csvwriter.writerow([i+1,self.csr_mat.indices[j]+1,self.csr_mat.data[j]])
        return

    def get_numbers(self):
        total = len(self.body["code"]) - self.last_pos - self.last_neg
        pos = Counter(self.body["code"])["yes"] - self.last_pos
        neg = Counter(self.body["code"])["no"] - self.last_neg
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def export(self):
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "code","time"]
        with open("../workspace/coded/" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)
            ## sort before export
            time_order = np.argsort(self.body["time"])[::-1]
            yes = [c for c in time_order if self.body["code"][c]=="yes"]
            no = [c for c in time_order if self.body["code"][c] == "no"]
            und = [c for c in time_order if self.body["code"][c] == "undetermined"]
            ##
            for ind in yes+no+und:
                csvwriter.writerow([self.body[field][ind] for field in fields])
        return

    def preprocess(self):
        ### Combine title and abstract for training ###########
        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore")
        tfidf = tfidfer.fit_transform(content)
        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        self.voc = np.array(tfidfer.vocabulary_.keys())[np.argsort(tfidfer.vocabulary_.values())][kept]
        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(content)
        ########################################################
        return

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","w") as handle:
            pickle.dump(self,handle)

    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "r") as handle:
            tmp = pickle.load(handle)
        return tmp

    def estimate_curve(self,clf):
        ## estimate ##
        # self.est_num=Counter(clf.predict(self.csr_mat[self.pool]))["yes"]
        from sklearn import linear_model
        import random


        def prob_sample(probs):
            order=np.argsort(probs)[::-1]
            count=0
            can=[]
            sample=[]
            for i,x in enumerate(probs[order]):
                count=count+x
                can.append(order[i])
                if count>=1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count=0
                    can=[]
            return sample

        # def prob_sample(probs):
        #     order=np.argsort(probs)[::-1]
        #     count=0
        #     can=[]
        #     sample=[]
        #     where = 1
        #     for i,x in enumerate(probs[order]):
        #         count=count+x
        #         can.append(order[i])
        #         if count>=where:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             where=where+1
        #             can=[]
        #     return sample

        # def prob_sample(probs):
        #     sample=[]
        #     for i,x in enumerate(probs):
        #         if random.random<x:
        #             sample.append(i)
        #     return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]


        decayed = list(poses) + list(negs)
        y = [1] * len(poses) + [0] * len(negs)
        prob = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob])

        es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)

        es.fit(prob[decayed], y)
        pos_at = list(es.classes_).index(1)
        pre = es.predict_proba(prob)[:, pos_at]
        est1 = len(poses)-self.last_pos + sum(pre[self.pool])
        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob])

        y=np.array([1 if x=='yes' else 0 for x in self.body['code']])
        y0=np.copy(y)


        es = linear_model.LogisticRegression(penalty='l2',fit_intercept =True)

        pos_num_last=Counter(y)[1]

        lifes=3
        life=lifes
        pos_num=Counter(y)[1]
        while (True):
            es.fit(prob, y)
            pos_at = list(es.classes_).index(1)
            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            y=np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            if pos_num == pos_num_last:
                life=life-1
                if life==0:
                    break
            else:
                life=lifes
            pos_num_last = pos_num
        esty=pos_num-self.last_pos
        self.record_est['x'].append(len(poses)+len(negs)-self.last_pos-self.last_neg)
        self.record_est['semi'].append(esty)
        self.record_est['sigmoid'].append(est1)
        pre = es.predict_proba(prob)[:, pos_at]

        ###
        pre2 = es.predict_proba(prob[self.pool])[:, pos_at]
        y = np.copy(y0)
        for x in self.pool[np.argsort(pre2)[-pos_num+len(poses):]]:
            y[x]=1
        es.fit(prob, y)
        pos_at = list(es.classes_).index(1)
        pre2 = es.predict_proba(prob)[:, pos_at]
        ###

        ##### simu curve #######
        # self.simcurve={'x':[self.record['x'][-1]],'pos':[self.record['pos'][-1]]}
        # already=decayed
        # pool=np.where(np.array(self.body['code']) == "undetermined")[0]
        # clff=svm.SVC(kernel='linear', probability=True)
        # while True:
        #     clff.fit(self.csr_mat[already], y[already])
        #     pos_at = list(clff.classes_).index(1)
        #     prob = clff.predict_proba(self.csr_mat[pool])[:, pos_at]
        #     sample = pool[np.argsort(prob)[::-1][:self.step]]
        #     already = already+list(sample)
        #     pool = np.array(list(set(pool)-set(sample)))
        #     self.simcurve['x'].append(self.simcurve['x'][-1]+self.step)
        #     self.simcurve['pos'].append(Counter(y[already])[1])
        #     if self.simcurve['pos'][-1] > int(Counter(y)[1]*0.9) or self.simcurve['pos'][-1]==self.simcurve['pos'][-2]:
        #         break
        # set_trace()
        ########################

        return esty,pre,pre2



    ## Train model ##
    def train(self,pne=False,weighting=False):
        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),self.atleast)),replace=False)
        except:
            pass

        if not pne:
            unlabeled=[]

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code']])
        all_neg=list(negs)+list(unlabeled)
        all = list(decayed)+list(unlabeled)
        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        # est_num, self.est, self.est2 = self.estimate_curve(clf)
        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id[:self.step], certain_prob[:self.step], certain_id

    ## reuse
    def train_reuse(self,pne=True):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []



        decayed = list(left) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed), self.atleast)), replace=False)
        except:
            pass

        if not pne:
            unlabeled = []


        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        est_num, probs = self.estimate_curve(clf)
        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, probs[uncertain_id], certain_id, probs[certain_id]

    ## not in use currently
    def train_reuse_random(self):
        thres=50

        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        pos, neg, total = self.get_numbers()
        if pos == 0 or pos + neg < thres:
            left=poses
            decayed = list(left) + list(negs)
        else:
            left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
            negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]
            decayed = list(left)+list(negs)
        clf.fit(self.csr_mat[decayed], np.array(self.body['code'])[decayed])
        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[negs])
            negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(left)]
            sample = list(left) + list(negs[negs_sel])
            clf.fit(self.csr_mat[sample], np.array(self.body['code'])[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if pos == 0 or pos + neg < thres:
            return uncertain_id, uncertain_prob, np.random.choice(list(set(certain_id) | set(self.random())),
                                                                  size=np.min((self.step, len(
                                                                      set(certain_id) | set(self.random())))),
                                                                  replace=False), certain_prob
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob



    ## Train_kept model ##
    def train_kept(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        ## only use latest poses
        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[::-1][:self.kept]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[poses])[::-1][:self.kept]]
        decayed = list(left)+list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed),self.atleast)), replace=False)
        except:
            pass

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## not in use currently
    def train_pos(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = poses
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []

        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),self.atleast)),replace=False)
        except:
            pass

        # print("%d,%d,%d" %(len(left),len(negs),len(unlabeled)))

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result

    ## Code candidate studies ##
    def code(self,id,label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()
        plt.plot(self.record['x'], self.record["pos"])
        ### estimation ####
        if Counter(self.body['code'])['yes']>=self.enough:
            est=self.est2[self.pool]
            order=np.argsort(est)[::-1]
            xx=[self.record["x"][-1]]
            yy=[self.record["pos"][-1]]
            for x in xrange(int(len(order)/self.step)):
                delta = sum(est[order[x*self.step:(x+1)*self.step]])
                if delta>=0.1:
                    yy.append(yy[-1]+delta)
                    xx.append(xx[-1]+self.step)
                else:
                    break
            plt.plot(xx, yy, "-.")
        ####################
        plt.ylabel("Relevant Found")
        plt.xlabel("Documents Reviewed")
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name

    def get_allpos(self):
        return len([1 for c in self.body["label"] if c=="yes"])-self.last_pos

    ## Restart ##
    def restart(self):
        os.remove("./memory/"+self.name+".pickle")

    ## Get missed relevant docs ##
    def get_rest(self):
        rest=[x for x in xrange(len(self.body['label'])) if self.body['label'][x]=='yes' and self.body['code'][x]!='yes']
        rests={}
        # fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        fields = ["Document Title"]
        for r in rest:
            rests[r]={}
            for f in fields:
                rests[r][f]=self.body[f][r]
        set_trace()
        return rests

    def cache_est(self):
        est = self.est[self.pool]
        order = np.argsort(est)[::-1]
        xx = [self.record["x"][-1]]
        yy = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy.append(yy[-1] + delta)
                xx.append(xx[-1] + self.step)
            else:
                break
        self.xx=xx
        self.yy=yy

        est = self.est2[self.pool]
        order = np.argsort(est)[::-1]
        xx2 = [self.record["x"][-1]]
        yy2 = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy2.append(yy2[-1] + delta)
                xx2.append(xx2[-1] + self.step)
            else:
                break
        self.xx2 = xx2
        self.yy2 = yy2