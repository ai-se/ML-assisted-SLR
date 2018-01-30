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

    true=62
    total=7002



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
    plt.figure(1)
    for j, ind in enumerate(stats['pos']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats['pos'][ind]['x'], np.array(stats['pos'][ind]['pos'])/true)
    plt.ylabel(str(file).split("_")[1] + "\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.40), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/recall_" + str(file) + ".eps")
    plt.savefig("../figure/recall_" + str(file) + ".png")

    plt.figure(2)
    for j, ind in enumerate(stats['est']):
        # if ind == 50 or ind == 0 or ind==100:
        if ind == 50:
            plt.plot(stats['est'][ind]['x'], [true/total]*len(stats['est'][ind]['x']), linestyle=lines[0], label='true')
            index=1
            for key in stats['est'][ind]:
                if key=='x':
                    continue
                # plt.plot(stats['est'][ind]['x'], stats['est'][ind][key],linestyle=lines[index],label=key)
                plt.plot(stats['est'][ind]['x'], np.array(stats['est'][ind][key])/total, linestyle=lines[index], label=key)
                index=index+1

    plt.ylabel(str(file).split("_")[1] + "\nPrevalence")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/prev_" + str(file) + ".eps")
    plt.savefig("../figure/prev_" + str(file) + ".png")





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


def test_estimate(first):
    repeats=30
    result={'pos':[],'est':[]}
    for i in xrange(repeats):
        first = str(first)
        a = START(first)
        result['est'].append(a.record_est)
        result['pos'].append(a.record)
        print(i,end=" ")
    with open("../dump/est_"+first.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def test_wallace(first):
    repeats=30
    result={'pos':[],'est':[]}
    for i in xrange(repeats):
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
    stop=0.90
    thres = 40

    read = Wallace()
    read = read.create(filename)
    read.restart()
    read = Wallace()
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

def START_DOC2VEC(filename):
    stop=0.95
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
            a,b,c,d,e =read.train(weighting=True)
            for id in c:
                read.code(id, read.body["label"][id])
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

def Codes(filename, code):
    stop=0.95
    thres = 0
    if "P" in code:
        starting = 5
    else:
        starting = 1

    weighting = "W" in code or "M" in code
    uncertain = "U" in code
    stopping = "S" in code

    read = MAR()
    read = read.create(filename)
    read.restart()
    read = MAR()
    read = read.create(filename)
    if not ("A" in code or "M" in code):
        read.enough = 100000
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d, %d" %(pos,pos+neg))
        if pos >= target:
            break
        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,c,d,e =read.train(weighting=weighting)
            if pos < 30 and uncertain:
                for id in a:
                    read.code(id, read.body["label"][id])
            else:
                if stopping:
                    now = 0
                    while pos < target:
                        for id in e[now:now+read.step]:
                            read.code(id, read.body["label"][id])
                        pos, neg, total = read.get_numbers()
                        now=now+read.step
                else:
                    for id in c:
                        read.code(id, read.body["label"][id])
    return read

def run_Codes(filename):
    repeats = 30
    result={}
    result['linear']=[]
    for i in xrange(repeats):
        np.random.seed(i)
        read = LINEAR(filename)
        result['linear'].append(read.record)
    for code1 in ['P','H']:
        for code2 in ['U','C']:
            for code3 in ['S','T']:
                for code4 in ['A','W','M','N']:
                    code = code1+code2+code3+code4
                    result[code]=[]
                    for i in xrange(repeats):
                        np.random.seed(i)
                        read = Codes(filename,code)
                        result[code].append(read.record)
                        # print("%s: %d" %(code,i))
    with open("../dump/codes_"+filename.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def summary(filename):
    with open("../dump/"+str(filename)+".pickle", "r") as f:
        results=pickle.load(f)
    test=[]
    total = results['linear'][0]['x'][-1]
    wss95 = []
    for key in results:
        # if 'M' in key:
        #     continue
        tmp=[]
        tmp_wss = []
        for r in results[key]:
            if key == 'linear':
                tmp.append(int(r['x'][-1]*0.95))
                tmp_wss.append(0.05 - (total - int(r['x'][-1]*0.95)) / total)
            else:
                tmp.append(r['x'][-1])
                tmp_wss.append(0.05 - (total-r['x'][-1])/total)
        test.append([key]+tmp)
        wss95.append([key]+tmp_wss)
    rdivDemo(test,isLatex=True)
    set_trace()
    rdivDemo(wss95, isLatex=False)


def summary_chart():
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 30}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 28, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    fig = plt.figure()
    ax = plt.subplot(111)

    for id, filename in enumerate(['codes_Wahono','codes_Hall','codes_Danijel','codes_K_all3'][::-1]):
        with open("../dump/"+str(filename)+".pickle", "r") as f:
            results=pickle.load(f)
        test={}
        total = results['linear'][0]['x'][-1]
        wss95 = {}
        for key in results:
            # if 'M' in key:
            #     continue
            tmp=[]
            tmp_wss = []
            for r in results[key]:
                if key == 'linear':
                    tmp.append(int(r['x'][-1] * 0.95))
                    tmp_wss.append(0.05 - (total - int(r['x'][-1] * 0.95)) / total)
                else:
                    tmp.append(r['x'][-1])
                    tmp_wss.append(0.05 - (total-r['x'][-1])/total)
            test[key]=np.median(tmp)
            wss95[key] = -np.median(tmp_wss)
        # if id==0:
        #     wss95['HUSA'],wss95['HUTM']=wss95['HUTM'],wss95['HUSA']
        test2={}
        for key in test:
            thesize = 150
            if key=='HUTM':
                color = 'red'
                thesize = 600
            elif key =='PUSA':
                color = 'orange'
                thesize = 300
            elif key=='PCSW':
                color = 'blue'
                thesize = 300
            elif key=='HCTN':
                color = 'green'
                thesize = 300
            elif key=='linear':
                color = 'black'
                thesize = 300
            elif key == 'HUSM':
                color = 'black'
            else:
                color = 'gray'
            if color =='gray':
                ax.scatter([wss95[key]],[id+1],s=thesize,c=color)
            else:
                test2[key]=wss95[key]
        for key in ['HUTM','PUSA','PCSW','HCTN','linear','HUSM']:
            thesize = 150
            if key == 'HUTM':
                treatment = 'FASTREAD'
                color = 'red'
                thesize = 600
            elif key =='PUSA':
                treatment = 'Wallace\'10'
                color = 'orange'
                thesize = 300
            elif key=='PCSW':
                treatment = 'Miwa\'14'
                color = 'blue'
                thesize = 300
            elif key=='HCTN':
                treatment = 'Cormack\'14'
                color = 'green'
                thesize = 300
            elif key=='linear':
                treatment = 'Linear Review'
                color = 'black'
                thesize = 300
            elif key=='HUSM':
                treatment = 'Others'
                color = 'gray'
            if id==0:
                ax.scatter([test2[key]],[id+1],s=thesize,c=color, label = treatment)
            else:
                ax.scatter([test2[key]], [id + 1], s=thesize, c=color)
    plt.xlabel("WSS@95")
    plt.ylim((0, 5))
    # plt.xlim((0, 9000))
    y = range(5)
    # plt.legend(bbox_to_anchor=(0., 1.32, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0,
    #                  box.width, box.height * 0.7])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
    #           fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(top=0.75,left=0.2,bottom=0.2,right=0.95)
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3)


    ylabels = ['','Kitchenham','Radjenovic','Hall','Wahono']
    plt.yticks(y, ylabels)


    plt.savefig("../figure/codes_chart.eps")
    plt.savefig("../figure/codes_chart.png")


def draw_selected(file):
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
    line=[0,0,0,0,0]
    keys = ['HUTA', 'HUTW', 'HUTM']
    for i,key in enumerate(keys):
        for j,ind in enumerate(stats[key]):
            if ind==50 or ind==0 or ind==100:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[i],label=five[j]+"_"+str(key).capitalize(),color=colors[j])

    # plt.plot(stats['linear'][50]['x'], stats['linear'][50]['pos'], linestyle=lines[i+1],
    #          label=five[j] + "_" + str(key).capitalize(), color='green')


    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/"+str(file)+".eps")
    plt.savefig("../figure/"+str(file)+".png")

def draw_selected2(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)



    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']
    line=[0,0,0,0,0]
    keys = ['HUTA', 'HUTW', 'HUTM']
    what = 11
    for i,key in enumerate(keys):
        plt.plot(results[key][what]['x'], results[key][what]['pos'],linestyle=lines[i],label=str(key).capitalize())

    # plt.plot(stats['linear'][50]['x'], stats['linear'][50]['pos'], linestyle=lines[i+1],
    #          label=five[j] + "_" + str(key).capitalize(), color='green')


    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/"+str(file)+".eps")
    plt.savefig("../figure/"+str(file)+".png")


def MISSING(filename):
    with open("../workspace/data/" + str(filename), "r") as csvfile:
        content = [x for x in csv.reader(csvfile, delimiter=',')]
    field = 'content'
    header = content[0]
    ind = header.index(field)
    cont = [c[ind] for c in content[1:]]
    yes = np.where(np.array(cont) == "yes")[0]
    total = len(yes)
    print(total)
    repeats=30
    code = 'HUTM'

    rec = []
    for i in xrange(repeats):
        read = Codes(filename,code)
        yes_code = np.where(np.array(read.body['code']) == "yes")[0]
        incl = len(set(yes) & set(yes_code))
        rec.append(incl)
    print(rec)

def MISSING2(filename):
    with open("../workspace/data/" + str(filename), "r") as csvfile:
        content = [x for x in csv.reader(csvfile, delimiter=',')]
    field = 'abs'
    header = content[0]
    ind = header.index(field)
    cont = [c[ind] for c in content[1:]]
    yes = np.where(np.array(cont) == "yes")[0]
    total = len(yes)
    print(total)
    repeats=30
    code = 'HUTM'

    rec = []
    for i in xrange(repeats):
        read = Codes(filename,code)
        yes_code = np.where(np.array(read.body['code']) != "undetermined")[0]
        incl = len(set(yes) & set(yes_code))
        rec.append(incl)
    print(rec)

def blocks2():
    import re
    x = [
        "  1 &         HUTM &    670  &  230 & 0.85 & 0.04 \\1 &         HCTM &    740  &  220 & 0.84 & 0.03 \\\hline  2 &         HUTA &    780  &  140 & 0.84 & 0.02 \\2 &         HCTW &    790  &  90 & 0.84 & 0.02 \\2 &         HUTW &    800  &  110 & 0.84 & 0.02 \\2 &         HCTA &    800  &  140 & 0.83 & 0.02 \\\hline  3 &         PCTM &    1150  &  450 & 0.78 & 0.07  \\3 &         PUTM &    1180  &  420 & 0.78 & 0.07 \\3 &         PCTA &    1190  &  340 & 0.78 & 0.05 \\3 &         PUTA &    1190  &  340 & 0.78 & 0.05 \\3&         PCTW &    1210  &  350 &  0.78 & 0.06 \\3 &         PUTW &    1220  &  370 &  0.77 & 0.06 \\\hline  4 &         HUSM &    1410  &  400 & 0.75 & 0.06 \\\hline  5 &         HUSA &    1610  &  370 & 0.72 & 0.07 \\\hline  6 &         PUSM &    1810  &  370 & 0.69 & 0.06 \\6 &         PUSA &    1910  &  700 & 0.67 & 0.10 \\\hline  7 &         HUSW &    2220  &  400 & 0.63 & 0.06 \\7 &         PUSW &    2240  &  360 & 0.63 & 0.06 \\\hline  8 &         HUTN &    2700  &  40 & 0.56 & 0.01 \\8 &         HCTN &    2720  &  40 & 0.56 & 0.01 \\8 &         PCSW &    2860  &  1320 & 0.54 & 0.20 \\8 &         PCSM &    2860  &  1320 & 0.54 & 0.20 \\8 &         PCTN &    2850  &  1130 & 0.54 & 0.17 \\8 &         PUTN &    2850  &  1130 & 0.54 & 0.17 \\\hline  9 &         PCSN &    3020  &  1810 & 0.51 & 0.26 \\9 &         PCSA &    3020  &  1810 & 0.51 & 0.26 \\\hline 10 &         HUSN &    4320  &  110 &  0.33 & 0.03 \\10 &         PUSN &    4370  &  1290 & 0.32 & 0.19 \\\hline 11 &       linear &    6650  &  0 & 0 & 0 \\11 &         HCSA &    6490  &  2760 & -0.01 & 0.39 \\ 11 &         HCSN &    6490  &  2760 & -0.01 & 0.39 \\11 &         HCSM &    6490  &  3110 & -0.01 & 0.44 \\11 &         HCSW &    6490  &  3110 & -0.01 & 0.44 \\",
        "  1 &         HUTW &    350  &  80 & 0.91 & 0.01 \\1 &         HUTA &    360  &  140 & 0.91 & 0.02 \\1 &         HUTM &    360  &  140 & 0.91 & 0.02 \\1 &         HCTW &    370  &  50 & 0.91 & 0.01 \\\hline  2 &         HCTM &    400  &  90 & 0.90 & 0.01 \\2 &         HCTA &    410  &  140 & 0.90 & 0.02 \\2 &         HUTN &    430  &  100 & 0.90 & 0.01 \\2 &         HCTN &    460  &  70 & 0.90 & 0.01 \\\hline  3 &         HUSM &    630  &  160 & 0.88 & 0.03 \\3 &         PCTW &    640  &  190 & 0.88 & 0.02 \\3 &         PUTW &    640  &  220 & 0.88 & 0.03 \\\hline  4 &         PCTN &    680  &  210 & 0.87 & 0.03 \\4 &         PUTN &    680  &  200 & 0.87 & 0.03 \\4 &         PUTM &    690  &  230 & 0.87 & 0.03 \\4 &         PCTA &    730  &  260 & 0.87 & 0.03 \\4 &         PCTM &    720  &  230 & 0.87 & 0.03 \\4 &         PUTA &    730  &  230 & 0.87 & 0.03 \\\hline  5 &         HUSW &    790  &  320 & 0.86 & 0.04 \\5 &         HUSA &    790  &  200 & 0.86 & 0.03 \\5 &         PUSW &    840  &  280 & 0.86 & 0.03 \\5 &         PUSM &    860  &  320 & 0.85 & 0.04 \\5 &         PUSA &    970  &  310 & 0.84 & 0.04 \\\hline  6 &         PCSW &    1560  &  580 & 0.77 & 0.07 \\6 &         PCSM &    1560  &  580 & 0.77 & 0.07 \\7 &         PUSN &    1680  &  1390 & 0.76 & 0.18 \\7 &         PCSN &    1990  &  690 & 0.72 & 0.09 \\7 &         PCSA &    1990  &  690 & 0.72 & 0.09 \\\hline  8 &         HUSN &    2270  &  1230 & 0.69 & 0.16 \\\hline  9 &         HCSA &    7500  &  5170 & 0.03 & 0.58 \\9 &         HCSN &    7500  &  5170 & 0.03 & 0.58 \\9 &       linear &    8464  &  0 & 0 & 0 \\9 &         HCSM &    8840  &  5340 & -0.04 & 0.60 \\9 &         HCSW &    8840  &  5340 & -0.04 & 0.60 \\",
        "  1 &         HUTM &    680   &  180  & 0.83 & 0.03 \\1 &         HCTM &    780   &  130  & 0.82 & 0.02 \\1 &         HCTA &    790   &  180  & 0.82 & 0.03 \\1 &         HUTA &    800   &  180  & 0.82 & 0.03 \\\hline  2 &         HUSA &    890   &  310  & 0.80 & 0.06 \\2 &         HUSM &    890   &  270  & 0.80 & 0.05 \\\hline  3 &         HUTW &    960   &  80  & 0.79 & 0.02 \\3 &         HCTW &    980   &  60  & 0.79 & 0.01 \\3 &         HUSW &    1080   &  410  & 0.77 & 0.07 \\\hline  4 &         PCTM &    1150   &  270  & 0.76 & 0.05 \\4 &         PUTM &    1150   &  270  & 0.76 & 0.05 \\\hline  5 &         HUTN &    1250   &  100  &  0.74 & 0.02 \\5 &         PCTA &    1260   &  210  & 0.74 & 0.05 \\5 &         PUTA &    1260   &  210  & 0.74 & 0.05 \\5 &         HCTN &    1270   &  70  & 0.74 & 0.02 \\5 &         PUSM &    1250   &  400  & 0.74 & 0.07 \\5 &         PUSW &    1250   &  450  & 0.73 & 0.08 \\5 &         PUTW &    1350   &  310  & 0.72 & 0.06 \\5 &         PCTW &    1370   &  310  & 0.72 & 0.06 \\5 &         PUSA &    1400   &  490  & 0.71 & 0.09 \\\hline  6 &         HUSN &    1570   &  300  & 0.69 & 0.05 \\6 &         PCTN &    1600   &  360  & 0.68 & 0.06 \\6 &         PUTN &    1600   &  360  & 0.68 & 0.06 \\\hline  7 &         PUSN &    1890   &  320  &  0.64 & 0.06 \\\hline8 &         PCSW &    2250   &  940  & 0.57 & 0.20 \\8 &         PCSM &    2250   &  940  & 0.57 & 0.20 \\\hline  9 &         PCSN &    2840   &  1680  & 0.47 & 0.31 \\9 &         PCSA &    2840   &  1680  & 0.47 & 0.31 \\\hline 10 &         HCSA &    5310   &  2140  & 0.07 & 0.36 \\10 &         HCSN &    5310   &  2140  & 0.07 & 0.36  \\10 &         HCSM &    5320   &  2200  &  0.02 & 0.37  \\10 &         HCSW &    5320   &  2200  & 0.02 & 0.37 \\",
        "  1 &         HUTM &    760  &  170 & 0.50 & 0.14 \\1 &         HUTA &    840  &  100 & 0.46 & 0.06 \\1 &         PUTM &    850  &  180 & 0.45 & 0.11 \\1 &         PCTM &    860  &  130 & 0.44 & 0.09 \\\hline  2 &         HCTA &    900  &  190 & 0.42 & 0.14 \\2 &         PCTA &    930  &  170 & 0.40 & 0.11 \\2 &         HCTM &    930  &  130 & 0.40 & 0.08 \\2 &         PUTA &    930  &  170 & 0.40 & 0.11 \\\hline  3 &         PUSW &    1140  &  250 & 0.27 & 0.15 \\3 &         PUSM &    1140  &  250 & 0.27 & 0.15 \\3 &         HUTW &    1160  &  10 & 0.27 & 0.01 \\3 &         HCTW &    1180  &  40 & 0.25 & 0.03 \\3 &         PCTW &    1190  &  170 & 0.25 & 0.10 \\3 &         PUTW &    1190  &  170 & 0.25 & 0.10 \\\hline  4 &         HUSW &    1200  &  150 & 0.24 & 0.10 \\4 &         HUSM &    1200  &  150 & 0.24 & 0.10 \\4 &         HUSN &    1250  &  220 & 0.21 & 0.14 \\4 &         HUSA &    1250  &  220 & 0.21 & 0.14 \\4 &         PUSA &    1250  &  290 & 0.21 & 0.19 \\4 &         PUSN &    1250  &  290 & 0.21 & 0.19 \\4 &         HUTN &    1260  &  10 & 0.21 & 0.01 \\4 &         HCTN &    1280  &  30 & 0.20 & 0.02 \\4 &         PUTN &    1280  &  260 & 0.19 & 0.16 \\4 &         PCTN &    1290  &  260 & 0.19 & 0.15 \\\hline  5 &         PCSW &    1370  &  260 & 0.14 & 0.22 \\5 &         PCSM &    1370  &  260 & 0.14 & 0.22 \\5 &         PCSN &    1400  &  340 & 0.12 & 0.22 \\5 &         PCSA &    1400  &  340 & 0.12 & 0.22 \\\hline  6 &       linear &    1615  &  0 & 0 & 0 \\\hline  7 &         HCSA &    1670  &  90 & -0.03 & 0.05 \\7 &         HCSN &    1670  &  90 & -0.03 & 0.05 \\7 &         HCSM &    1670  &  90 & -0.03 & 0.05 \\7 &         HCSW &    1670  &  90 & -0.03 & 0.05 \\"]

    x=map( lambda a: re.sub('\hline','',a), x)
    x = map( lambda a: re.sub(' ', '', a).split('\\'), x)
    new = map( lambda b: np.array([a.split('&') for a in b if a!=""]), x)

    titles = [n[:,1] for n in new]
    ranks = [n[:,0] for n in new]
    code0=['H','P']
    code1=['U','C']
    code2=['T','S']
    code3=['M','A','W','N']

    order0=''
    for c1 in code1:
        for c2 in code2:
            for c3 in code3:
                for c0 in code0:
                    code = c0+c1+c2+c3
                    tmp=[code] + [ranks[i][list(xx).index(code)] for i,xx in enumerate(titles)]
                    order0=order0+' & '.join(tmp)+'\\\\ \n'
                order0=order0+'\\hline \n'
    print(order0)
    set_trace()
    order1 = ''
    for c0 in code0:
        for c2 in code2:
            for c3 in code3:
                for c1 in code1:
                    code = c0 + c1 + c2 + c3
                    tmp = [code] + [ranks[i][list(xx).index(code)] for i, xx in enumerate(titles)]
                    order1 = order1 + ' & '.join(tmp) + '\\\\ \n'
                order1 = order1 + '\\hline \n'
    print(order1)
    set_trace()
    order2 = ''
    for c0 in code0:
        for c1 in code1:
            for c3 in code3:
                for c2 in code2:
                    code = c0 + c1 + c2 + c3
                    tmp = [code] + [ranks[i][list(xx).index(code)] for i, xx in enumerate(titles)]
                    order2 = order2 + ' & '.join(tmp) + '\\\\ \n'
                order2 = order2 + '\\hline \n'
    print(order2)
    set_trace()
    order3 = ''
    for c0 in code0:
        for c1 in code1:
            for c2 in code2:
                for c3 in code3:
                    code = c0 + c1 + c2 + c3
                    tmp = [code] + [ranks[i][list(xx).index(code)] for i, xx in enumerate(titles)]
                    order3 = order3 + ' & '.join(tmp) + '\\\\ \n'
                order3 = order3 + '\\hline \n'
    print(order3)
    set_trace()

def blocks():
    import re
    x="  1 &         HUTW &    350  &  80 & 0.91 & 0.01 \\1 &         HUTA &    360  &  140 & 0.91 & 0.02 \\1 &         HUTM &    360  &  140 & 0.91 & 0.02 \\1 &         HCTW &    370  &  50 & 0.91 & 0.01 \\\hline  2 &         HCTM &    400  &  90 & 0.90 & 0.01 \\2 &         HCTA &    410  &  140 & 0.90 & 0.02 \\2 &         HUTN &    430  &  100 & 0.90 & 0.01 \\2 &         HCTN &    460  &  70 & 0.90 & 0.01 \\\hline  3 &         HUSM &    630  &  160 & 0.88 & 0.03 \\3 &         PCTW &    640  &  190 & 0.88 & 0.02 \\3 &         PUTW &    640  &  220 & 0.88 & 0.03 \\\hline  4 &         PCTN &    680  &  210 & 0.87 & 0.03 \\4 &         PUTN &    680  &  200 & 0.87 & 0.03 \\4 &         PUTM &    690  &  230 & 0.87 & 0.03 \\4 &         PCTA &    730  &  260 & 0.87 & 0.03 \\4 &         PCTM &    720  &  230 & 0.87 & 0.03 \\4 &         PUTA &    730  &  230 & 0.87 & 0.03 \\\hline  5 &         HUSW &    790  &  320 & 0.86 & 0.04 \\5 &         HUSA &    790  &  200 & 0.86 & 0.03 \\5 &         PUSW &    840  &  280 & 0.86 & 0.03 \\5 &         PUSM &    860  &  320 & 0.85 & 0.04 \\5 &         PUSA &    970  &  310 & 0.84 & 0.04 \\\hline  6 &         PCSW &    1560  &  580 & 0.77 & 0.07 \\6 &         PCSM &    1560  &  580 & 0.77 & 0.07 \\7 &         PUSN &    1680  &  1390 & 0.76 & 0.18 \\7 &         PCSN &    1990  &  690 & 0.72 & 0.09 \\7 &         PCSA &    1990  &  690 & 0.72 & 0.09 \\\hline  8 &         HUSN &    2270  &  1230 & 0.69 & 0.16 \\\hline  9 &         HCSA &    7500  &  5170 & 0.03 & 0.58 \\9 &         HCSN &    7500  &  5170 & 0.03 & 0.58 \\9 &       linear &    8464  &  0 & 0 & 0 \\9 &         HCSM &    8840  &  5340 & -0.04 & 0.60 \\9 &         HCSW &    8840  &  5340 & -0.04 & 0.60 \\"
    x=re.sub('\hline','',x)
    x = re.sub(' ', '', x)
    xx = x.split('\\')
    new = np.array([a.split('&') for a in xx if a!=""])
    title = new[:,1]
    code0=['H','P']
    code1=['U','C']
    code2=['T','S']
    code3=['M','A','W','N']

    order0=''
    for c1 in code1:
        for c2 in code2:
            for c3 in code3:
                for c0 in code0:
                    code = c0+c1+c2+c3
                    order0=order0+' & '.join(new[list(title).index(code)])+'\\\\ \n'
                order0=order0+'\\hline \n'
    print(order0)
    set_trace()
    order1 = ''
    for c0 in code0:
        for c2 in code2:
            for c3 in code3:
                for c1 in code1:
                    code = c0 + c1 + c2 + c3
                    order1 = order1 + ' & '.join(new[list(title).index(code)]) + '\\\\ \n'
                order1 = order1 + '\\hline \n'
    print(order1)
    set_trace()
    order2 = ''
    for c0 in code0:
        for c1 in code1:
            for c3 in code3:
                for c2 in code2:
                    code = c0 + c1 + c2 + c3
                    order2 = order2 + ' & '.join(new[list(title).index(code)]) + '\\\\ \n'
                order2 = order2 + '\\hline \n'
    print(order2)
    set_trace()
    order3 = ''
    for c0 in code0:
        for c1 in code1:
            for c2 in code2:
                for c3 in code3:
                    code = c0 + c1 + c2 + c3
                    order3 = order3 + ' & '.join(new[list(title).index(code)]) + '\\\\ \n'
                order3 = order3 + '\\hline \n'
    print(order3)
    set_trace()


##################
def ERROR(filename):
    read = MAR()
    read = read.create(filename)
    read.lda()
    read.syn_error()

if __name__ == "__main__":
    eval(cmd())
