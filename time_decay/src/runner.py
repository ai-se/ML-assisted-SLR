from __future__ import division, print_function

import csv
import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sk import rdivDemo
import unicodedata
import random
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





"normalization_row"
def normalize(mat,ord=2):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,ord=ord)
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

def use_or_not(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    with open("../dump/"+str(file)+"0.pickle", "r") as f:
        results0=pickle.load(f)


    stats=bestNworst(results)
    stats0 = bestNworst(results0)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    plt.figure()
    for key in stats0:
        a = key.split("_")[0]
        if a=="start":
            for j,ind in enumerate(stats0[key]):
                plt.plot(stats0[key][ind]['x'], stats0[key][ind]['pos'],linestyle=lines[0],color=colors[j],label=five[j]+"_no")
    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            for j,ind in enumerate(stats[key]):
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[1],color=colors[j],label=five[j]+"_yes")


    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/"+str(file).split('_')[1]+".eps")
    plt.savefig("../figure/"+str(file).split('_')[1]+".png")


def draw(file):
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
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='FASTREAD'
        try:
            b = key.split("_")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind == 50 or ind == 100:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel("Study Retrieval")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
        plt.savefig("../figure/"+str(file)+str(i)+".eps")
        plt.savefig("../figure/"+str(file)+str(i)+".png")

def update_median_draw(file):
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
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]
    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='FASTREAD'
        try:
            b = key.split("_")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind==50:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel("Retrieval Rate")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/median_"+str(file)+str(i)+".eps")
        plt.savefig("../figure/median_"+str(file)+str(i)+".png")

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


def exp():
    data = ["Hall.csv","Wahono.csv","Danijel.csv"]
    for a in data:
        for b in data:
            if a==b:
                continue
            else:
                # time_not(a,b)
                update_or_reuse(a,b)

def update_or_reuse(first,second):
    first = str(first)
    second = str(second)
    repeats=30
    result={"update":[],"reuse":[],"start":[],"update-reuse":[]}
    for i in xrange(repeats):
        a = START(first)
        a.export()

        e = UPDATE_REUSE(second, first)
        result["update-reuse"].append(e.record)
        e.restart()

        d = START(second)
        result["start"].append(d.record)
        d.restart()

        c = REUSE(second,first)
        result["reuse"].append(c.record)
        c.restart()



        b = UPDATE(second,first)
        result["update"].append(b.record)
        b.restart()
        a.restart()

        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/"+first.split('.')[0]+"_"+second.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def time_not(first,second):
    first = str(first)
    second = str(second)
    repeats=30
    result={"time":[],"start":[],"update-reuse":[]}
    for i in xrange(repeats):
        a = START(first)
        a.export()

        e = UPDATE_REUSE(second, first)
        result["update-reuse"].append(e.record)
        e.restart()

        d = START(second)
        result["start"].append(d.record)
        d.restart()

        a.restart()

        a = TIME_START(first)
        a.export()

        c = TIME(second,first)
        result["time"].append(c.record)
        c.restart()

        a.restart()

        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/time_"+first.split('.')[0]+"_"+second.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def update_exp():
    update("Hall2007-.csv","Hall2007+.csv","Hall.csv")
    update("Wahono2008-.csv","Wahono2008+.csv","Wahono.csv")
    update("Danijel2005-.csv","Danijel2005+.csv","Danijel.csv")

def update(first,second,all):
    first = str(first)
    second = str(second)
    repeats=30
    result={"POS_2":[],"UPDATE_2":[],"FASTREAD_2":[],"FASTREAD_1":[],"FASTREAD_0":[]}
    for i in xrange(repeats):
        a = START(first)
        a.export()
        result["FASTREAD_1"].append(a.record)

        b = POS(second,first)
        result["POS_2"].append(b.record)
        b.restart()

        c = UPDATE(second,first)
        result["UPDATE_2"].append(c.record)
        c.restart()

        d = START(second)
        result["FASTREAD_2"].append(d.record)
        d.restart()

        e = START(all)
        result["FASTREAD_0"].append(e.record)
        e.restart()

        a.restart()
        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/UPDATE_"+all.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def simple(first):
    first = str(first)
    a = START_AUTO(first)
    a.export()
    a.plot()

def START(filename):
    stop=0.90

    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d, %d" %(pos,pos+neg))
        if pos >= target:
            break
        if pos==0:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def TIME(filename,old):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        a,b,ids,c =read.train_kept()
        for id in ids:
            read.code(id, read.body["label"][id])
    return read

def TIME_START(filename):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        if pos==0:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train_kept()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def UPDATE(filename,old,pne=False):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        a,b,ids,c =read.train(pne)
        for id in ids:
            read.code(id, read.body["label"][id])
    return read

def POS(filename,old):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        a,b,ids,c =read.train_pos()
        for id in ids:
            read.code(id, read.body["label"][id])
    return read


def REUSE_RANDOM(filename,old):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        a,b,ids,c =read.train_reuse_random()
        for id in ids:
            read.code(id, read.body["label"][id])
    return read

def REUSE(filename,old):
    stop=0.9
    thres=5

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        if pos < thres:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
        else:
            a, b, ids, c = read.train_reuse()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def UPDATE_REUSE(filename,old):
    stop=0.9
    lifes=2
    life=lifes
    last_pos=0
    thres=5

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos, pos + neg))

        if pos-last_pos:
            life=lifes
        else:
            life=life-1
        last_pos=pos


        if pos >= target:
            break
        # if (pos >= thres or pos==0) and life<1:
        if (pos >= thres) and life<1:
            # print("reuse")
            lifes=0
            a,b,ids,c =read.train_reuse()
            for id in ids:
                read.code(id, read.body["label"][id])
        else:
            # print("update")
            a, b, ids, c = read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def START_AUTO(filename):
    read = MAR()
    read = read.create(filename)
    pos_last=0
    full_life=3
    life=full_life
    while True:
        pos, neg, total = read.get_numbers()
        print("%d/ %d" % (pos,pos+neg))
        if pos >= 10:
            if pos==pos_last:
                life=life-1
                if life==0:
                    break
            else:
                life=full_life
        if pos==0:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
        pos_last=pos
    return read

if __name__ == "__main__":
    eval(cmd())
