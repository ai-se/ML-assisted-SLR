from __future__ import division, print_function


import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sk import rdivDemo
import unicodedata
from sklearn import svm
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import lda
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from mar import MAR




def colorcode(N):
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=N-1, clip=True)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap




"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat


def hcca_lda(csr_mat,csr_lda, labels, step=10 ,initial=10, pos_limit=1, thres=30, stop=0.9):
    num=len(labels)
    pool=range(num)
    train=[]
    steps = np.array(range(int(num / step))) * step

    pos=0
    pos_track=[0]
    clf = svm.SVC(kernel='linear', probability=True)
    begin=0
    result={}
    enough=False

    total=Counter(labels)["yes"]*stop

    # total = 1000

    for idx, round in enumerate(steps[:-1]):

        if round >= 2500:
            if enough:
                pos_track_f=pos_track9
                train_f=train9
                pos_track_l=pos_track8
                train_l=train8
            elif begin:
                pos_track_f=pos_track4
                train_f=train4
                pos_track_l=pos_track2
                train_l=train2
            else:
                pos_track_f=pos_track
                train_f=train
                pos_track_l=pos_track
                train_l=train
            break

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
            pool4 = pool2[:]
            train4 = train2[:]
            pos_track4 = pos_track2[:]
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

            ## lda
            clf.fit(csr_lda[train2], labels[train2])
            pred_proba2 = clf.predict_proba(csr_lda[pool2])
            pos_at = list(clf.classes_).index("yes")
            proba2 = pred_proba2[:, pos_at]
            sort_order_certain2 = np.argsort(1 - proba2)
            can2 = [pool2[i] for i in sort_order_certain2[:step]]
            train2.extend(can2)
            pool2 = list(set(pool2) - set(can2))
            pos = Counter(labels[train2])["yes"]
            pos_track2.append(pos)


            ################ new *_C_C_A
            if not enough:
                if pos>=thres:
                    enough=True
                    pos_track9=pos_track4[:]
                    train9=train4[:]
                    pool9=pool4[:]
                    pos_track8=pos_track2[:]
                    train8=train2[:]
                    pool8=pool2[:]
            else:
                clf.fit(csr_mat[train9], labels[train9])
                poses = np.where(labels[train9] == "yes")[0]
                negs = np.where(labels[train9] == "no")[0]
                train_dist = clf.decision_function(csr_mat[train9][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample9 = np.array(train9)[poses].tolist() + np.array(train9)[negs][negs_sel].tolist()

                clf.fit(csr_mat[sample9], labels[sample9])
                pred_proba9 = clf.predict_proba(csr_mat[pool9])
                pos_at = list(clf.classes_).index("yes")
                proba9 = pred_proba9[:, pos_at]
                sort_order_certain9 = np.argsort(1 - proba9)
                can9 = [pool9[i] for i in sort_order_certain9[:step]]
                train9.extend(can9)
                pool9 = list(set(pool9) - set(can9))
                pos = Counter(labels[train9])["yes"]
                pos_track9.append(pos)

                clf.fit(csr_lda[train8], labels[train8])
                poses = np.where(labels[train8] == "yes")[0]
                negs = np.where(labels[train8] == "no")[0]
                train_dist = clf.decision_function(csr_lda[train8][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample8 = np.array(train8)[poses].tolist() + np.array(train8)[negs][negs_sel].tolist()

                clf.fit(csr_lda[sample8], labels[sample8])
                pred_proba8 = clf.predict_proba(csr_lda[pool8])
                pos_at = list(clf.classes_).index("yes")
                proba8 = pred_proba8[:, pos_at]
                sort_order_certain8 = np.argsort(1 - proba8)
                can8 = [pool8[i] for i in sort_order_certain8[:step]]
                train8.extend(can8)
                pool8 = list(set(pool8) - set(can8))
                pos = Counter(labels[train8])["yes"]
                pos_track8.append(pos)

        print("Round #{id} passed\r".format(id=round), end="")

    result["begin"] = begin
    result["x"] = steps[:len(pos_track_f)]
    result["new_continuous_aggressive"] = pos_track_f
    result["lda"] = pos_track_l
    return result, train_f



##### draw

def update_repeat_draw(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)


    stats=bestNworst(results)
    colors=['blue','purple','green','brown','red']
    lines=['-','--',':']
    five=['best','$Q_1$','median','$Q_3$','worst']


    line=[0,0,0,0]
    for key in stats:
        a = key.split("_")[0]
        b = key.split("_")[1]
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a))
        line[int(b)]+=1

    for i in xrange(3):
        plt.figure(i+1)
        plt.ylabel("Retrieval Rate")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=i+1, borderaxespad=0.)
        plt.savefig("../figure/"+str(file)+str(i+1)+".eps")
        plt.savefig("../figure/"+str(file)+str(i+1)+".png")

def bestNworst(results):
    stats={}

    for key in results:
        stats[key]={}
        result=results[key]
        order = np.argsort([r['x'][-1] for r in result])
        for ind in [0,25,50,75,100]:
            stats[key][ind]=result[order[int(ind*(len(order)-1)/100)]]

    return stats

##### UPDATE exp
def update_exp():
    repeats=30
    result={"start_1":[],"start_2":[], "start_3":[],"update_2":[],"update_3":[],"reuse_3":[]}
    for i in xrange(repeats):
        a = START("Hall2007-.csv")
        result["start_1"].append(a.record)
        a.export()

        b = START("Hall2007+.csv")
        result["start_2"].append(b.record)
        b.restart()

        c = UPDATE("Hall2007+.csv","Hall2007-.csv")
        result["update_2"].append(c.record)
        c.export()

        d = START("Wahono.csv")
        result["start_3"].append(d.record)
        d.restart()

        e = UPDATE("Wahono.csv","Hall2007+.csv")
        result["update_3"].append(e.record)
        e.restart()

        f = REUSE("Wahono.csv",c)
        result["reuse_3"].append(f.record)
        f.restart()
        c.restart()
        # print("Repeat #{id} finished\r".format(id=i), end="")
        print(i, end=" ")
    with open("../dump/everything.pickle","wb") as handle:
        pickle.dump(result,handle)

def update_or_reuse():
    repeats=30
    result={"update":[],"reuse":[]}
    for i in xrange(repeats):
        a = START("Hall.csv")
        a.export()

        c = REUSE("Wahono.csv",a)
        result["reuse"].append(c.record)
        c.restart()
        b = UPDATE("Wahono.csv","Hall.csv")
        result["update"].append(b.record)
        b.restart()
        a.restart()
        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/update_or_reuse.pickle","wb") as handle:
        pickle.dump(result,handle)


def START(filename):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        if pos > target:
            break
        if pos==0:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def UPDATE(filename,old):
    stop=0.9

    read = MAR()
    read = read.create_UPDATE(filename,old)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        if pos > target:
            break
        a,b,ids,c =read.train()
        for id in ids:
            read.code(id, read.body["label"][id])
    return read

def UPDATE_ALL(filename,old):
    stop=0.9

    read = MAR()
    read = read.create_UPDATE_ALL(filename,old)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        if pos > target:
            break
        a,b,ids,c =read.train()
        for id in ids:
            read.code(id, read.body["label"][id])
    return read



def model_transform(model,vocab,vocab_new):
    w=[]
    for term in vocab_new:
        try:
            ind=vocab.index(term)
            w.append(model['w'][0,ind])
        except:
            w.append(0)
    model['w']=csr_matrix(w)
    return model

def REUSE(filename,old):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    model = model_transform({'w':old.get_clf().coef_, "pos_at":list(old.get_clf().classes_).index("yes")},old.voc,read.voc)
    while True:
        pos, neg, total = read.get_numbers()
        if pos > target:
            break
        if pos==0 or pos+neg<50:
            for id in read.reuse(model):
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def similarity(a,b):
    tops=30

    read = MAR()
    read = read.create(a)
    body_a = [read.body["Document Title"][index] + " " + read.body["Abstract"][index] for index in
                   xrange(len(read.body["Document Title"]))]
    label_a = read.body['label']
    read = read.create(b)
    body_b = [read.body["Document Title"][index] + " " + read.body["Abstract"][index] for index in
                   xrange(len(read.body["Document Title"]))]
    label_b = read.body['label']
    body_c = body_a+body_b

    tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False)
    tf_c = tfer.fit_transform(body_c).astype(np.int32)
    tf_a = tfer.transform(body_a).astype(np.int32)
    tf_b = tfer.transform(body_b).astype(np.int32)

    clt = lda.LDA(n_topics=tops, n_iter=200, alpha=0.8, eta=0.8)
    dis = csr_matrix(clt.fit_transform(tf_c))
    dis_a = dis[:tf_a.shape[0]]
    dis_b = dis[tf_a.shape[0]:]

    sum_a = dis_a.sum(axis=0)/dis_a.shape[0]
    sum_b = dis_b.sum(axis=0)/dis_b.shape[0]
    x=range(tops)


    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    plt.figure()
    plt.plot(x, np.array(sum_a)[0] ,label=a.split('.')[0])
    plt.plot(x, np.array(sum_b)[0] ,label=b.split('.')[0])


    plt.ylabel("Topic Weight")
    plt.xlabel("Topic ID")
    plt.legend(bbox_to_anchor=(0.9, 0.90), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/data_"+a.split('.')[0]+" vs "+b.split('.')[0]+".eps")
    plt.savefig("../figure/data_"+a.split('.')[0]+" vs "+b.split('.')[0]+".png")


    # Data similarity
    score = sum([min((sum_a[0,i],sum_b[0,i])) for i in xrange(tops)])
    print("data: %f" %score)

    # Target similarity
    pos_a = [i for i in xrange(len(label_a)) if label_a[i]=='yes']
    pos_b = [i for i in xrange(len(label_b)) if label_b[i]=='yes']

    dis_pos_a = dis_a[pos_a]
    dis_pos_b = dis_b[pos_b]
    sum_pos_a = dis_pos_a.sum(axis=0)/dis_pos_a.shape[0]
    sum_pos_b = dis_pos_b.sum(axis=0)/dis_pos_b.shape[0]

    plt.figure()
    plt.plot(x, np.array(sum_pos_a)[0] ,label=a.split('.')[0])
    plt.plot(x, np.array(sum_pos_b)[0] ,label=b.split('.')[0])


    plt.ylabel("Topic Weight")
    plt.xlabel("Topic ID")
    plt.legend(bbox_to_anchor=(0.9, 0.90), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/target_"+a.split('.')[0]+" vs "+b.split('.')[0]+".eps")
    plt.savefig("../figure/target_"+a.split('.')[0]+" vs "+b.split('.')[0]+".png")

    score2 = sum([min((sum_pos_a[0,i],sum_pos_b[0,i])) for i in xrange(tops)])
    print("target: %f" %score2)
    set_trace()








## Start rule (clustering)
def init_sample(data,n_clusters,samples):
    cluster=KMeans(n_clusters=n_clusters)
    cluster.fit(data)
    result=cluster.labels_
    x=list(set(result))
    pool=[]
    for key in x:
        a=[i for i in xrange(data.shape[0])if result[i]==key]
        pool.extend(list(np.random.choice(a,samples,replace=False)))
    return pool

if __name__ == "__main__":
    eval(cmd())
