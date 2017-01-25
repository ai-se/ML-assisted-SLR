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







##### draw

def bestNworst(results):
    stats={}

    for key in results:
        stats[key]={}
        result=results[key]
        order = np.argsort([r['x'][-1] for r in result])
        for ind in [0,25,50,75,100]:
            stats[key][ind]=result[order[int(ind*(len(order)-1)/100)]]

    return stats


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
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    for key in stats:
        a = key.split(":")[0]
        if a=="start":
            a='FASTREAD'
        try:
            b = key.split(":")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind==50 or ind==100:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel("Retrieval Rate")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
        plt.savefig("../figure/"+str(file)+str(i)+".eps")
        plt.savefig("../figure/"+str(file)+str(i)+".png")


##### UPDATE exp

def reuse_exp():
    data = ["Hall.csv","Wahono.csv","Abdellatif.csv"]
    for a in data:
        for b in data:
            if a==b:
                continue
            else:
                update_or_reuse(a,b)

def update_or_reuse(first,second):
    first = str(first)
    second = str(second)
    repeats=30
    result={"update":[],"reuse":[],"start":[],"update_reuse":[]}
    for i in xrange(repeats):
        a = START(first)
        a.export()

        d = START(second)
        result["start"].append(d.record)
        d.restart()

        c = REUSE_RANDOM(second,first)
        result["reuse"].append(c.record)
        c.restart()

        b = UPDATE(second,first)
        result["update"].append(b.record)
        b.restart()

        e = UPDATE_REUSE(second, first)
        result["update_reuse"].append(e.record)
        e.restart()
        a.restart()

        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/"+first.split('.')[0]+"_"+second.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)


def time_exp(first,second):
    result={}
    # a = START(first)
    # a.plot()
    # a.export()

    b = UPDATE_REUSE(second, first)
    result["update_reuse"] = b.record
    b.restart()

    b = REUSE_RANDOM(second, first)
    result["reuse_random"] = b.record
    b.restart()

    b = REUSE(second, first)
    result["reuse"] = b.record
    b.restart()

    b = UPDATE(second, first)
    result["update"] = b.record
    b.restart()

    b = TIME(second, first)
    result["time"] = b.record
    b.restart()
    print(result)
    # a.restart()


def START(filename):
    stop=0.90

    read = MAR()
    read = read.create(filename)
    num = read.get_allpos()
    target = int(num*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" %(pos,pos+neg))
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

def UPDATE(filename,old):
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
        a,b,ids,c =read.train()
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
        if pos >0 and life<1:
            a,b,ids,c =read.train_reuse()
            for id in ids:
                read.code(id, read.body["label"][id])
        else:
            a, b, ids, c = read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
    return read



if __name__ == "__main__":
    eval(cmd())
