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

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from mar import MAR





#### export

def export(file):
    read = MAR()
    read = read.create_lda(file)
    read.export_feature()

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

def stats(file):

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    name=file.split('_')[0]
    test=[]
    for key in results:
        a=key.split('_')[0]
        try:
            b=key.split('_')[1]
        except:
            b='0'
        if b=="2" and name=="UPDATE":
            if a=="POS":
                a="UPDATE_POS"
            elif a=="UPDATE":
                a="UPDATE_ALL"
            tmp=[]
            for r in results[key]:
                tmp.append(r['x'][-1])
            print(a+": max %d" %max(tmp))
            test.append([a]+tmp)
        elif name!="UPDATE":
            tmp=[]
            for r in results[key]:
                tmp.append(r['x'][-1])
            test.append([a]+tmp)
            print(a+": max %d" %max(tmp))
    rdivDemo(test,isLatex=True)
    set_trace()

def pro_simple(first):

    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/rest_"+first+".pickle","rb") as handle:
        rest=pickle.load(handle)
    coun = Counter(sum(rest.keys(), []))
    order = np.argsort(coun.values())[::-1]
    x= np.array(coun.keys())[order]
    y= np.array(coun.values())[order]
    xx = range(len(x))

    plt.figure()
    plt.plot(xx, y)
    plt.ylabel("Number of times left")
    plt.xlabel("Candidate ID")
    plt.xticks(xx, x)
    plt.savefig("../figure/left_"+str(first)+".eps")
    plt.savefig("../figure/left_"+str(first)+".png")





def draw(file):
    font = {'family': 'normal',
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
    five=['$0th$','$25th$','$50th$','$75th$','$100th$']

    nums = set([])
    line=[0,0,0,0,0]

    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='FASTREAD'
        if a=="POS":
            a='UPDATE_POS'
        if a=="UPDATE":
            a='UPDATE_ALL'
        try:
            b = key.split("_")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))

        if key=="linear":
            plt.plot(stats[key][50]['x'], stats[key][50]['pos'],linestyle=lines[line[int(b)]],color='black',label="Linear Review")
        else:
            for j,ind in enumerate(stats[key]):
                if ind == 50 or ind == 75 or ind==25:
                    plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel(str(file).split("_")[1]+"\nRelevant Studies")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.40), loc=1, ncol=2, borderaxespad=0.)
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

def exp2():
    update_or_reuse("Hall2007-.csv","Hall2007+.csv")
    update_or_reuse("Wahono2008-.csv","Wahono2008+.csv")
    update_or_reuse("Danijel2005-.csv","Danijel2005+.csv")

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
    repeats=30
    rest=[]
    for i in xrange(repeats):
        first = str(first)
        a = START(first)
        tmp=a.get_rest()
        rest.append(tmp)
    with open("../dump/rest_"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(rest,handle)

def repeat_auto(first):
    repeats=30
    rec={"linear":[],"fastread":[]}
    for i in xrange(repeats):
        first = str(first)
        a = START_AUTO(first)
        rec['fastread'].append(a.record)
        a.restart()
        b=LINEAR(first)
        rec['linear'].append(b.record)
        b.restart()

    with open("../dump/"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(rec,handle)

## basic units


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

def START_ERROR(filename):
    read = MAR()
    read = read.create(filename)
    pos_last=0
    full_life=3
    human_error=0.2
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
                if read.body["label"][id]=="no":
                    if random.random()<human_error**2:
                        hl="yes"
                    else:
                        hl="no"
                elif read.body["label"][id]=="yes":
                    if random.random()<2*(human_error-human_error**2):
                        hl="no"
                    else:
                        hl="yes"
                read.code(id, hl)
        else:
            a,b,ids,c =read.train()
            for id in ids:
                if read.body["label"][id]=="no":
                    if random.random()<human_error**2:
                        hl="yes"
                    else:
                        hl="no"
                elif read.body["label"][id]=="yes":
                    if random.random()<2*(human_error-human_error**2):
                        hl="no"
                    else:
                        hl="yes"
                read.code(id, hl)
        pos_last=pos
    read.export()
    return read

def LINEAR(filename):
    read = MAR()
    read = read.create(filename)
    while True:
        pos, neg, total = read.get_numbers()
        if total-(pos+neg)<10:
            break
        for id in read.random():
            read.code(id, read.body["label"][id])
    return read


if __name__ == "__main__":
    eval(cmd())
