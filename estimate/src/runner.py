from __future__ import division, print_function

import csv
import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sk import rdivDemo,a12slow
import unicodedata
import random
from sklearn import svm
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from mar import MAR
from wallace import Wallace





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


def draw_est(file):
    which=str(file)
    if which=="Wahono":
        true=62
        total=7002
    elif which =="Hall":
        true = 106
        total = 8991
    elif which =="Danijel":
        true = 48
        total = 6000
    elif which =="K_all3":
        true = 45
        total = 1704
    else:
        true = 45
        total = 1704

    np.random.seed(0)
    uniform = [1]*true+[0]*(total-true)
    np.random.shuffle(uniform)
    uniform_test = [sum(uniform[:where*10]) for where in range(int(total/10)+1)]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (12, 6)}
    plt.rcParams.update(paras)

    with open("../dump/est_"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    with open("../dump/wallace_"+str(file)+".pickle", "r") as f:
        results2=pickle.load(f)


    stats=bestNworst(results)
    stats2 = bestNworst(results2)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['$0th$','$25th$','$50th$','$75th$','$100th$']

    nums = set([])
    line=[0,0,0,0,0]
    plt.figure(1)
    for j, ind in enumerate(stats['pos']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats['pos'][ind]['x'], np.array(stats['pos'][ind]['pos'])/true, linestyle=lines[0], label='FASTREAD')
            plt.plot(stats2['pos'][ind]['x'], np.array(stats2['pos'][ind]['pos']) / true, linestyle=lines[1], label="sampling $\\propto$ probabilities")
            plt.plot(stats['pos'][ind]['x'], np.array(uniform_test)[(np.array(stats['pos'][ind]['x'])/10).astype(int)] / true, linestyle=lines[2], label='uniform random sampling')
    plt.ylabel("Recall")
    plt.xlabel("# Studies Reviewed")
    plt.legend(bbox_to_anchor=(1, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/recall_" + str(file) + ".eps")
    plt.savefig("../figure/recall_" + str(file) + ".png")

    plt.figure(2)
    for j, ind in enumerate(stats['est']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats['est'][ind]['x'], [true/total]*len(stats['est'][ind]['x']), linestyle=lines[0], label='true')

            plt.plot(stats['est'][ind]['x'], np.array(uniform_test)[(np.array(stats['est'][ind]['x'])/10).astype(int)].astype(float) / np.array(stats['est'][ind]['x']) , linestyle=lines[1], label='estimated (uniform random sampling)')
            index=2
            for key in stats['est'][ind]:
                if key=='x':
                    continue
                if key=='sigmoid':
                    name='probability prediction'
                elif key=="semi":
                    name='SEMI'
                else:
                    name=key
                # plt.plot(stats['est'][ind]['x'], stats['est'][ind][key],linestyle=lines[index],label=key)
                plt.plot(stats['est'][ind]['x'], np.array(stats['est'][ind][key])/total, linestyle=lines[index], label="estimated ("+name+")")
                index=index+1

    plt.ylabel("Prevalence")
    plt.xlabel("# Studies Reviewed")
    plt.legend(bbox_to_anchor=(1, 0.9), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/prev_est_" + str(file) + ".eps")
    plt.savefig("../figure/prev_est_" + str(file) + ".png")

    plt.figure(3)
    for j, ind in enumerate(stats2['est']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats2['est'][ind]['x'], [true / total] * len(stats2['est'][ind]['x']), linestyle=lines[0],
                     label='true')
            plt.plot(stats['est'][ind]['x'],
                     np.array(uniform_test)[(np.array(stats['est'][ind]['x']) / 10).astype(int)].astype(
                         float) / np.array(stats['est'][ind]['x']), linestyle=lines[1],
                     label='estimated (uniform random sampling)')

            index = 2
            for key in stats2['est'][ind]:
                if key == 'x':
                    continue
                if key=='sigmoid':
                    name='probability prediction'
                elif key=="est":
                    name='Wallace adjusted'
                    continue
                else:
                    name=key
                # plt.plot(stats['est'][ind]['x'], stats['est'][ind][key],linestyle=lines[index],label=key)
                plt.plot(stats2['est'][ind]['x'], np.array(stats2['est'][ind][key]) / total, linestyle=lines[index],
                         label="estimated ("+name+")")
                index = index + 1

    plt.ylabel("Prevalence")
    plt.xlabel("# Studies Reviewed")
    plt.legend(bbox_to_anchor=(1, 0.9), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/prev_wallace_" + str(file) + ".eps")
    plt.savefig("../figure/prev_wallace_" + str(file) + ".png")

def draw_est2(file):
    which=str(file)
    if which=="Wahono":
        true=62
        total=7002
    elif which =="Hall":
        true = 104
        total = 8991
    elif which =="Danijel":
        true = 48
        total = 6000
    elif which =="K_all3":
        true = 45
        total = 1704
    else:
        true = 45
        total = 1704

    np.random.seed(0)
    uniform = [1]*true+[0]*(total-true)
    np.random.shuffle(uniform)
    uniform_test = [sum(uniform[:where*10]) for where in range(int(total/10)+1)]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 28, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/est_"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    with open("../dump/wallace_"+str(file)+".pickle", "r") as f:
        results2=pickle.load(f)


    stats=bestNworst(results)
    stats2 = bestNworst(results2)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['$0th$','$25th$','$50th$','$75th$','$100th$']

    nums = set([])
    line=[0,0,0,0,0]
    plt.figure(1)

    ax = plt.subplot(111)
    for j, ind in enumerate(stats['pos']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            ax.plot(stats['pos'][ind]['x'], np.array(uniform_test)[(np.array(stats['pos'][ind]['x'])/10).astype(int)] / true, color='red', linestyle=lines[1], label='uniform \nrandom \nsampling')
            ax.plot(stats['pos'][ind]['x'], np.array(stats['pos'][ind]['pos'])/true, linestyle=lines[2], color='blue', label="SEMI")
            xx = [stats2['pos'][ind]['x'][i] for i in xrange(len(stats2['pos'][ind]['x'])) if stats2['pos'][ind]['x'][i]<1500]
            yy = [stats2['pos'][ind]['pos'][i] for i in xrange(len(stats2['pos'][ind]['x'])) if
                  stats2['pos'][ind]['x'][i] < 1500]
            ax.plot(xx, np.array(yy) / true, linestyle=lines[3],
                    color='green', label="Wallace'13")

            # ax.plot(stats2['pos'][ind]['x'], np.array(stats2['pos'][ind]['pos']) / true, linestyle=lines[3], color='green', label="Wallace'13")
    plt.ylabel("Recall")
    plt.xlabel("#Papers Reviewed")
    plt.xlim(0, stats['pos'][ind]['x'][-1])
    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.75)

    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)

    plt.savefig("../figure/recall_all_" + str(file) + ".eps")
    plt.savefig("../figure/recall_all_" + str(file) + ".png")


    plt.figure(2)
    ax = plt.subplot(111)
    for j, ind in enumerate(stats['est']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats['est'][ind]['x'], [true/total]*len(stats['est'][ind]['x']), linestyle=lines[0], color='gray', label='true')

            plt.plot(stats['est'][ind]['x'], np.array(uniform_test)[(np.array(stats['est'][ind]['x'])/10).astype(int)].astype(float) / np.array(stats['est'][ind]['x']) , linestyle=lines[1], color='red', label='uniform \nrandom \nsampling')
            index=2
            for key in stats['est'][ind]:
                if key=="semi":
                    name='SEMI'
                # plt.plot(stats['est'][ind]['x'], stats['est'][ind][key],linestyle=lines[index],label=key)
                    plt.plot(stats['est'][ind]['x'], np.array(stats['est'][ind][key])/total, linestyle=lines[index], color='blue', label=name)
                    startpoint= stats['est'][ind]['x'][0]
                    index=index+1

    for j, ind in enumerate(stats2['est']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            for key in stats2['est'][ind]:
                if key=="est":
                    name="Wallace'13"
                    startind=2
                    xx = [stats2['est'][ind]['x'][i] for i in xrange(len(stats2['est'][ind]['x'])) if
                          stats2['est'][ind]['x'][i] < 1500]
                    yy = [stats2['est'][ind]['est'][i] for i in xrange(len(stats2['est'][ind]['x'])) if
                          stats2['est'][ind]['x'][i] < 1500]
                    ax.plot(xx[startind:], np.array(yy[startind:]) / total, linestyle=lines[index],
                            color='green', label="Wallace'13")
                    # plt.plot(stats2['est'][ind]['x'][startind:], np.array(stats2['est'][ind][key][startind:]) / total, color='green', linestyle=lines[index],
                    #          label=name)
                    index = index + 1

    plt.ylabel("Prevalence")
    plt.xlabel("#Papers Reviewed")
    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.75)

    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)
    # plt.show()
    plt.savefig("../figure/prev_all_" + str(file) + ".eps")
    plt.savefig("../figure/prev_all_" + str(file) + ".png")



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
                # if ind == 50 or ind == 75 or ind==25:
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
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],label=five[j]+"_"+str(a).capitalize())
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

        if results[key]==[]:
            continue
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
        # a = START(first)
        # a.export()
        # result["FASTREAD_1"].append(a.record)

        # b = POS(second,first)
        # result["POS_2"].append(b.record)
        # b.restart()

        c = UPDATE(second,first)
        result["UPDATE_2"].append(c.record)
        c.restart()

        d = START(second)
        result["FASTREAD_2"].append(d.record)
        d.restart()

        # e = START(all)
        # result["FASTREAD_0"].append(e.record)
        # e.restart()
        #
        # a.restart()
        print("Repeat #{id} finished\r".format(id=i), end="")
        # print(i, end=" ")
    with open("../dump/UPDATE_"+all.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def rest(first):
    repeats=30
    rest=[]
    for i in xrange(repeats):
        first = str(first)
        a = START(first)
        tmp=a.get_rest()
        rest.append(tmp)
    with open("../dump/rest_"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(rest,handle)


def test_estimate(first, query):
    repeats=1
    result={'pos':[],'est':[]}
    for i in xrange(repeats):
        np.random.seed(i+2)
        first = str(first)
        a = BM25(first, query, 'est', i + 2)
        result['est'].append(a.record_est)
        result['pos'].append(a.record)
        print(i,end=" ")
    with open("../dump/est_"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def test_wallace(first):
    repeats=1
    result={'pos':[],'est':[]}
    for i in xrange(repeats):
        np.random.seed(i+2)
        first = str(first)
        a = START_Wallace(first)
        result['est'].append(a.record_est)
        result['pos'].append(a.record)
        print(i,end=" ")
    with open("../dump/wallace_"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

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


def one_cache_est(filename):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    read = START_est(filename)

    plt.figure()
    plt.plot(read.record['x'], read.record['pos'], label="Actual Curve")
    try:
        plt.plot(read.xx, read.yy, '-.',label="Estimated Curve 1")
        plt.plot(read.xx2, read.yy2, '--', label="Estimated Curve 2")
    except:
        pass
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.savefig("../figure/cache_" + str(filename).split('.')[0] + ".eps")
    plt.savefig("../figure/cache_" + str(filename).split('.')[0] + ".png")


## basic units

def START_Wallace(filename):
    thres = 0

    read = Wallace()
    read = read.create(filename)
    target = 2000
    while True:
        pos, neg, total = read.get_numbers()
        print("%d, %d" %(pos,pos+neg))
        if pos+neg >= target or pos+neg >= total:
            break
        if pos==0 or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train(pne=True)
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def START(filename):
    stop=0.90
    thres = 40

    read = MAR()
    read = read.create(filename)
    read.restart()
    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        print("%d, %d" %(pos,pos+neg))
        if pos >= target:
            break
        if pos==0 or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train(weighting=True)
            for id in ids:
                read.code(id, read.body["label"][id])
    return read

def START_est(filename):
    stop=0.90
    thres = 40
    flag = True

    read = MAR()
    read = read.create(filename)
    read.restart()
    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d, %d" %(pos,pos+neg))
        if pos >= target:
            break
        if pos==0 or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train(pne=True)

            if pos >= 60 and flag:
                read.cache_est()
                # read.xx=read.simcurve['x']
                # read.yy=read.simcurve['pos']
                flag= False

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

def UPDATE(filename,old,pne=True):
    stop=0.9

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        print("%d/ %d" % (pos,pos+neg))
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

def REUSE(filename,old,pne=True):
    stop=0.9
    thres=5

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stop)
    while True:
        pos, neg, total = read.get_numbers()
        print("%d/ %d" % (pos,pos+neg))
        if pos >= target:
            break
        if pos < thres:
            a,b,ids,c =read.train(pne)
            for id in ids:
                read.code(id, read.body["label"][id])
        else:
            a, b, ids, c = read.train_reuse(pne)
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
    full_life=5
    life=full_life
    while True:
        pos, neg, total = read.get_numbers()
        print("%d/ %d" % (pos,pos+neg))
        if pos==0 or pos+neg<40:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,ids,c =read.train()
            for id in ids:
                read.code(id, read.body["label"][id])
            if pos==pos_last:
                life=life-1
                if life==0:
                    break
            else:
                life=full_life
        pos_last=pos
    return read

def UPDATE_AUTO(filename,old,pne=True):

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    pos_last=-1
    full_life=5
    life=full_life
    while True:
        pos, neg, total = read.get_numbers()
        print("%d/ %d" % (pos,pos+neg))
        if pos == pos_last:
            life = life - 1
            if life == 0:
                break
        else:
            life = full_life
        a,b,ids,c =read.train(pne)
        for id in ids:
            read.code(id, read.body["label"][id])
        pos_last = pos
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

### BM25
def BM25(filename, query, stop='true', seed=0):
    stopat = 10
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    stops = 1500

    read = MAR()
    read = read.create(filename)


    read.BM25(query.strip().split('_'))

    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total or pos + neg >= stops:
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code(id,read.body['label'][id])
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code(id, read.body['label'][id])
            else:
                for id in c:
                    read.code(id, read.body['label'][id])
    # read.export()
    # results = analyze(read)
    # print(results)
    return read

###################################
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
####################################
def Code_noError(filename, code):
    target=2000
    thres = 0
    if "P" in code:
        starting = 5
    else:
        starting = 1

    weighting = "W" in code or "M" in code
    uncertain = "U" in code

    read = MAR()
    read = read.create(filename)
    read.restart()
    read = MAR()
    read = read.create(filename)
    if not ("A" in code or "M" in code):
        read.enough = 100000
    while True:
        pos, neg, total = read.get_numbers()
        if pos+neg >= target or pos+neg >= total:
            break
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" % (pos, pos + neg))
        # if pos >= target:
        #     break
        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,c,d =read.train(weighting=weighting,pne=False)

            if pos < 30 and uncertain:
                for id in a:
                    read.code(id, read.body["label"][id])
            else:
                for id in c:
                    read.code(id, read.body["label"][id])
    return read



if __name__ == "__main__":
    eval(cmd())
