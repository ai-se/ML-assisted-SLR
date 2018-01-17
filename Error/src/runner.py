from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt

from sk import rdivDemo

import random

from collections import Counter

from mar import MAR
from wallace import Wallace



# from scipy.sparse import csr_matrix
# from sklearn.cluster import KMeans
#
# from time import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm
# import unicodedata
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# from sk import a12slow
# import csv



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
        set_trace()
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


def test_estimate(first,query):
    repeats = 1
    result = {'pos': [], 'est': []}
    for i in xrange(repeats):
        np.random.seed(i + 2)
        first = str(first)
        a = BM25(first, query, 'est', i+2)
        result['est'].append(a.record_est)
        result['pos'].append(a.record)
        print(i, end=" ")
    with open("../dump/est_" + first.split('.')[0] + ".pickle", "wb") as handle:
        pickle.dump(result, handle)

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
    stop=0.95
    thres = 40

    read = Wallace()
    read = read.create(filename)
    read.restart()
    read = Wallace()
    read = read.create(filename)
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
                        print("%s: %d" %(code,i))
    with open("../dump/codes_"+filename.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(result,handle)

def summary(filename):
    with open("../dump/"+str(filename)+".pickle", "r") as f:
        results=pickle.load(f)

    for rs in results:
        print(rs)
        test = {}
        for key in results[rs]:
            # if 'M' in key:
            #     continue
            tmp=[]
            for r in results[rs][key]:
                tmp.append(r['x'][-1])
            test[key]=tmp
        rdivDemo(test,isLatex=False)
        set_trace()

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
    keys = ['HCTA', 'HCTW', 'HCTM']
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


##################
def ERROR(filename):
    read = MAR()
    read = read.create(filename)
    read.lda()
    read.syn_thres = 0.9
    read.syn_error()

def Code_Error(filename, code):
    stop=0.95
    thres = 40
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
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" % (pos, pos + neg))
        # if pos >= target:
        #     break
        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code_error(id, read.body["label"][id])
        else:
            a,b,c,d =read.train(weighting=weighting)
            if read.est_num*stop<pos:
                break
            if pos < 30 and uncertain:
                for id in a:
                    read.code_error(id, read.body["label"][id])
            else:
                for id in c:
                    read.code_error(id, read.body["label"][id])
    read.export()
    return read


def Code_noError_repeats(filename, code="HUTM"):
    repeats=30
    record = [Code_noError(filename, code).record for i in xrange(repeats)]
    with open("../dump/stop0_"+filename.split('.')[0]+".pickle","wb") as handle:
        pickle.dump(record,handle)


def sum_result(filename):
    with open("../dump/"+filename+".pickle","rb") as handle:
        record = pickle.load(handle)
    result={'x':[],'pos':[]}
    for r in record:
        result['x'].append(r['x'][-1])
        result['pos'].append(r['pos'][-1])
    set_trace()


#############################

def UPDATE_ALL(filename,old,stop='true',pne=False,seed=0):
    np.random.seed(seed)
    stopat=0.95

    read = MAR()
    read = read.create(filename)
    read.create_old(old)
    num2 = read.get_allpos()
    target = int(num2*stopat)
    if stop =='est':
        read.enable_est=True
    else:
        read.enable_est = False
    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" % (pos, pos + neg))
        a, b, c, d = read.train(pne=pne)
        if stop=='est':
            if stopat*read.est_num <= pos:
                break
        else:
            if pos >= target:
                break

        if pos+read.last_pos < 30:
            for id in a:
                read.code(id, read.body["label"][id])
        else:
            for id in c:
                read.code(id, read.body["label"][id])
    return read

def UPDATE_POS(filename,old,stop='true',pne=True,seed=0):
    stopat=0.95
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)
    read.create_pos(old)
    num2 = read.get_allpos()
    target = int(num2*stopat)
    if stop =='est':
        read.enable_est=True
    else:
        read.enable_est = False
    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" % (pos, pos + neg))
        a, b, c, d = read.train(pne=pne)
        if stop=='est':
            if stopat*read.est_num <= pos:
                break
        else:
            if pos >= target:
                break

        if pos+read.last_pos < 30:
            for id in a:
                read.code(id, read.body["label"][id])
        else:
            for id in c:
                read.code(id, read.body["label"][id])
    return read


def REUSE(filename,old,stop='true',pne=True,seed=0):

    np.random.seed(seed)
    stopat = 0.95
    thres=10
    read = MAR()
    read = read.create(filename)
    read.create_pos(old)
    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False
    while True:
        pos, neg, total = read.get_numbers()
        # print("%d/ %d" % (pos,pos+neg))

        if pos < thres:
            a,b,c,d =read.train(pne)

            if pos+read.last_pos < 30:
                for id in a:
                    read.code(id, read.body["label"][id])
            else:
                for id in c:
                    read.code(id, read.body["label"][id])
        else:
            a, b, c, d = read.train_reuse(pne)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code(id, read.body["label"][id])
            else:
                for id in c:
                    read.code(id, read.body["label"][id])
    return read


def Code_noError(filename, code, stop='true',seed=0):
    np.random.seed(seed)
    stopat = 0.95
    thres = 0
    if "P" in code:
        starting = 5
    else:
        starting = 1

    weighting = "W" in code or "M" in code
    uncertain = "U" in code

    read = MAR()
    read = read.create(filename)
    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop =='est':
        read.enable_est=True
    else:
        read.enable_est = False
    if not ("A" in code or "M" in code):
        read.enough = 100000
    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" % (pos, pos + neg))

        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,c,d =read.train(weighting=weighting,pne=False)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'budget':
                if pos+neg>=1000:
                    break
            else:
                if pos >= target:
                    break
            if pos < 10 and uncertain:
                for id in a:
                    read.code(id, read.body["label"][id])
            else:
                for id in c:
                    read.code(id, read.body["label"][id])
    return read

def Auto_Rand(filename, stop='true', error='none', interval = 100000, seed=0):
    stopat = 0.95
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    read.interval = interval



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

        if pos + neg >= total:
            if stop == 'knee' and error == 'random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos + neg < thres:
            read.code_error(read.one_rand()[0], error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos > 0 and pos_last == pos:
                    counter = counter + 1
                else:
                    counter = 0
                pos_last = pos
                if counter >= 5:
                    break
            elif stop == 'knee':
                if pos > 0:
                    if read.knee():
                        if error == 'random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1 | part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    return read

def exp_BM25(stop='true'):
    repeats=30

    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction',
               "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    results={}
    for file in files:
        results[file]={}
        pos = []
        cost = []
        for i in xrange(repeats):
            read = BM25(file, queries[file], stop=stop, seed=i)
            pos.append(read.record['pos'][-1])
            cost.append(read.record['x'][-1])
        results[file]['pos'] = np.median(pos)
        results[file]['cost'] = np.median(cost)
    # print(results)
    with open("../dump/other_"+stop+".pickle","wb") as handle:
        pickle.dump(results,handle)
    set_trace()

def exp_result(stop='true'):
    with open("../dump/other_"+stop+".pickle","r") as handle:
        results = pickle.load(handle)
    print(results)

### BM25
def BM25(filename, query, stop='true', error='none', interval = 100000, seed=0):
    stopat = 0.95
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    read.interval = interval

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

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    return read

def analyze(read):
    unknown = np.where(np.array(read.body['code']) == "undetermined")[0]
    pos = np.where(np.array(read.body['code']) == "yes")[0]
    neg = np.where(np.array(read.body['code']) == "no")[0]
    yes = np.where(np.array(read.body['label']) == "yes")[0]
    no = np.where(np.array(read.body['label']) == "no")[0]
    falsepos = len(set(pos) & set(no))
    truepos = len(set(pos) & set(yes))
    falseneg = len(set(neg) & set(yes))
    unknownyes = len(set(unknown) & set(yes))
    unique = len(read.body['code']) - len(unknown)
    count = sum(read.body['count'])
    return {"falsepos": falsepos, "truepos": truepos, "falseneg": falseneg, "unknownyes": unknownyes, "unique": unique, "count": count}


############ scenarios ##########
def has_data(stop='true'):
    repeats=30
    datasets=[('Hall2007+.csv','Hall2007-.csv'),('Wahono2008+.csv','Wahono2008-.csv'),('Danijel2005+.csv','Danijel2005-.csv'),('K_all3+.csv','K_all3-.csv')]
    treatments=['UPDATE_POS','REUSE','Auto_Syn','BM25','RANDOM']
    treatments=['UPDATE_POS','BM25','RANDOM']
    results={}
    for data in datasets:
        results[data[0]]={}
        for treatment in treatments:
            if data[0]=='K_all3+.csv' and treatment=='UPDATE_ALL':
                continue
            results[data[0]][treatment]=[]

            for i in xrange(repeats):
                if treatment=="UPDATE_ALL":
                    read = UPDATE_ALL(data[0],data[1],stop=stop)
                elif treatment=="UPDATE_POS":
                    read = UPDATE_POS(data[0], data[1], stop=stop)
                elif treatment=="REUSE":
                    read = REUSE(data[0], data[1], stop=stop)
                elif treatment=="Auto_Syn":
                    if data[0]=="Hall2007+.csv":
                        syn_data='Syn_Hall.csv'
                    elif data[0]=="Wahono2008+.csv":
                        syn_data = 'Syn_Wahono.csv'
                    elif data[0]=="Danijel2005+.csv":
                        syn_data = 'Syn_Danijel.csv'
                    elif data[0]=="K_all3+.csv":
                        syn_data = 'Syn_Kitchenham.csv'
                    read = REUSE(data[0], syn_data, stop=stop)
                elif treatment=="BM25":
                    if data[0]=="Hall2007+.csv":
                        query='defect prediction'
                    elif data[0]=="Wahono2008+.csv":
                        query = 'defect prediction'
                    elif data[0]=="Danijel2005+.csv":
                        query = 'defect prediction metrics'
                    elif data[0]=="K_all3+.csv":
                        query = 'systematic review'
                    read = BM25(data[0], query=query, stop=stop)
                elif treatment=="RANDOM":
                    read = Code_noError(data[0],"HUTM" , stop=stop)
                results[data[0]][treatment].append(read.record)
                # read.restart()
                # print(data[0]+'_'+treatment+str(i),end=" ")
    with open("../dump/data_"+stop+".pickle","wb") as handle:
        pickle.dump(results,handle)

def no_data(stop='true'):
    repeats=30
    datasets=['Hall.csv','Wahono.csv','Danijel.csv','K_all3.csv']
    # datasets = ['Hall.csv']
    treatments = ['Auto_Syn', 'BM25', 'RANDOM', 'Cormack_BM25', 'Auto_Rand']
    results={}
    for data in datasets:
        results[data]={}
        for treatment in treatments:
            if data=='K_all3.csv' and (treatment=='REUSE' or treatment=="UPDATE_ALL" or treatment=="UPDATE_POS"):
                continue
            elif data == "Hall.csv":
                syn_data = 'Wahono.csv'
            elif data == "Wahono.csv":
                syn_data = 'Hall.csv'
            elif data == "Danijel.csv":
                syn_data = 'Hall.csv'
            results[data][treatment]=[]
            for i in xrange(repeats):
                if treatment=="UPDATE_ALL":
                    read = UPDATE_ALL(data,syn_data,stop=stop, seed=i)
                elif treatment=="UPDATE_POS":
                    read = UPDATE_POS(data, syn_data, stop=stop, seed=i)
                elif treatment=="REUSE":

                    read = REUSE(data, syn_data, stop=stop, seed=i)
                elif treatment=="Auto_Syn":
                    if data=="Hall.csv":
                        syn_data='Syn_Hall.csv'
                    elif data=="Wahono.csv":
                        syn_data = 'Syn_Wahono.csv'
                    elif data=="Danijel.csv":
                        syn_data = 'Syn_Danijel.csv'
                    elif data=="K_all3.csv":
                        syn_data = 'Syn_Kitchenham.csv'
                    read = REUSE(data, syn_data, stop=stop, seed=i)
                elif treatment=="Cormack_BM25":
                    if data=="Hall.csv":
                        syn_data='BM25_Hall.csv'
                    elif data=="Wahono.csv":
                        syn_data = 'BM25_Wahono.csv'
                    elif data=="Danijel.csv":
                        syn_data = 'BM25_Danijel.csv'
                    elif data=="K_all3.csv":
                        syn_data = 'BM25_Kitchenham.csv'
                    read = REUSE(data, syn_data, stop=stop, seed=i)
                elif treatment=="Auto_Rand":
                    if data=="Hall.csv":
                        syn_data='BM25_Hall.csv'
                    elif data=="Wahono.csv":
                        syn_data = 'BM25_Wahono.csv'
                    elif data=="Danijel.csv":
                        syn_data = 'BM25_Danijel.csv'
                    elif data=="K_all3.csv":
                        syn_data = 'BM25_Kitchenham.csv'
                    read = Auto_Rand(data, stop=stop, seed=i)
                elif treatment=="BM25":
                    if data == "Hall.csv":
                        query='defect_prediction'
                    elif data == "Wahono.csv":
                        query = 'defect_prediction'
                    elif data == "Danijel.csv":
                        query = 'defect_prediction_metrics'
                    elif data == "K_all3.csv":
                        query = 'systematic_review'
                    read = BM25(data, query=query, stop=stop, seed=i)
                elif treatment=="RANDOM":
                    read = Code_noError(data,"HUTM" , stop=stop, seed=i)
                results[data][treatment].append(read.record)
                # read.restart()
    with open("../dump/nodata1_"+stop+".pickle","wb") as handle:
        pickle.dump(results,handle)

def sum_res(filename):
    with open("../dump/"+filename+".pickle","r") as handle:
        record = pickle.load(handle)
    new={}
    for dataset in record:
        new[dataset]={}
        for treatment in record[dataset]:
            new[dataset][treatment]= [x['x'][-1] for x in record[dataset][treatment]]
        print(dataset)
        rdivDemo(new[dataset], isLatex=False)
        set_trace()

def sum_pos_x(filename):
    with open("../dump/"+filename+".pickle","r") as handle:
        record = pickle.load(handle)
    new={}
    for dataset in record:
        new[dataset]={}
        for treatment in record[dataset]:
            new[dataset][treatment]= [np.median([x['x'][-1] for x in record[dataset][treatment]]), np.median([x['pos'][-1] for x in record[dataset][treatment]])]
            print("%s, %s: %d, %d" %(dataset,treatment,new[dataset][treatment][0],new[dataset][treatment][1]))
        set_trace()

def draw_one():

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)
    lines=['-','-','--','-.',':']
    five=['$0th$','$25th$','$50th$','$75th$','$100th$']

    with open("../dump/nodata_true.pickle","r") as handle:
        record = pickle.load(handle)

    what = record['Hall.csv']['RANDOM']
    order = np.argsort([r['x'][-1] for r in what])
    stats={}
    for ind in [0,25,50,75,100]:
        stats[ind]=what[order[int(ind*(len(order)-1)/100)]]



    plt.figure(0)
    for j,ind in enumerate(stats):
        if ind == 50 or ind == 75 or ind==0 or ind==100:
            plt.plot(stats[ind]['x'], np.array(stats[ind]['pos'])/106,linestyle=lines[j],label=five[j]+" Percentile")
    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")

    docnum = 8991
    x=[i*100 for i in xrange(10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.legend(bbox_to_anchor=(0.9, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile.eps")
    plt.savefig("../figure/percentile.png")

def draw_two():

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)
    lines=['-','-','--','-.',':']
    five=['$0th$','$25th$','$50th$','$75th$','$100th$']

    with open("../dump/nodata_est.pickle","r") as handle:
        record = pickle.load(handle)

    what = record['Hall.csv']['BM25']
    order = np.argsort([r['x'][-1] for r in what])
    stats={}
    for ind in [0,25,50,75,100]:
        stats[ind]=what[order[int(ind*(len(order)-1)/100)]]



    plt.figure(0)
    for j,ind in enumerate(stats):
        if ind == 50 or ind == 75 or ind==0 or ind==100:
            plt.plot(stats[ind]['x'], np.array(stats[ind]['pos'])/106,linestyle=lines[j],label=five[j]+" Percentile")
    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")

    docnum = 8991
    x=[i*100 for i in xrange(7)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.legend(bbox_to_anchor=(0.9, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_BM25.eps")
    plt.savefig("../figure/percentile_BM25.png")

    what = record['Hall.csv']['UPDATE_POS']
    order = np.argsort([r['x'][-1] for r in what])
    stats={}
    for ind in [0,25,50,75,100]:
        stats[ind]=what[order[int(ind*(len(order)-1)/100)]]



    plt.figure(1)
    for j,ind in enumerate(stats):
        if ind == 50 or ind == 75 or ind==0 or ind==100:
            plt.plot(stats[ind]['x'], np.array(stats[ind]['pos'])/106,linestyle=lines[j],label=five[j]+" Percentile")
    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")

    docnum = 8991
    x=[i*100 for i in xrange(7)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.legend(bbox_to_anchor=(0.9, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_UPDATE.eps")
    plt.savefig("../figure/percentile_UPDATE.png")


def draw_three():

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)
    lines=['-','--','-.',':']
    five=['$0th$','$25th$','median result','$75th$','worst result']
    colors=["blue",'brown', 'green', 'yellow', 'red']

    with open("../dump/nodata_est1.pickle","r") as handle:
        record = pickle.load(handle)

    what = record['Hall.csv']['BM25']
    with open("../dump/nodata_true.pickle","r") as handle:
        record1 = pickle.load(handle)

    what1 = record1['Hall.csv']['RANDOM']


    order = np.argsort([r['x'][-1] for r in what])
    stats={}
    for ind in [0,25,50,75,100]:
        stats[ind]=what[order[int(ind*(len(order)-1)/100)]]

    order1 = np.argsort([r['x'][-1] for r in what1])
    stats1={}
    for ind in [0,25,50,75,100]:
        stats1[ind]=what1[order1[int(ind*(len(order1)-1)/100)]]



    plt.figure(0)
    for j,ind in enumerate(stats):
        if ind == 50 or ind==100:
            plt.plot(stats[ind]['x'], np.array(stats[ind]['pos'])/106,linestyle=lines[1], color=colors[j], label="FAST$^2$ ("+str(five[j])+")")
    for j,ind in enumerate(stats1):
        if ind == 50 or ind==100:
            plt.plot(stats1[ind]['x'], np.array(stats1[ind]['pos'])/106,linestyle=lines[0], color=colors[j], label="FASTREAD ("+str(five[j])+")")
    plt.ylabel("Recall")
    plt.xlabel("#Papers Reviewed")

    docnum = 8991
    x=[i*100 for i in xrange(10)]

    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)
    plt.ylim((0, 1))
    plt.xlim((0, 900))

    plt.legend(bbox_to_anchor=(1, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_all.eps")
    plt.savefig("../figure/percentile_all.png")


def sum_median_worst():
    with open("../dump/data_true.pickle","r") as handle:
        record = pickle.load(handle)
    with open("../dump/nodata_true.pickle","r") as handle:
        record1 = pickle.load(handle)
    dataname={"Danijel.csv": "Radjenovi{\\'c} (Full)", "Wahono.csv": "Wahono (Full)", "Hall.csv": "Hall (Full)", "K_all3.csv": "Kitchenham (Full)", "Danijel2005+.csv": "Radjenovi{\\'c} (Half)", "Wahono2008+.csv": "Wahono (Half)", "Hall2007+.csv": "Hall (Half)", "K_all3+.csv": "Kitchenham (Half)"}
    datasize={"Danijel.csv": 6000, "Wahono.csv": 7002, "Hall.csv": 8991, "K_all3.csv": 1704, "Danijel2005+.csv": 3035, "Wahono2008+.csv": 3810, "Hall2007+.csv": 4066, "K_all3+.csv": 1688}
    treatmentname={"Auto_Syn": "Auto-Syn", "BM25": "Auto-BM25", "RANDOM": "RANDOM", "UPDATE_POS": "UPDATE", "REUSE": "REUSE"}
    new={}
    for d in record:
        dataset=dataname[d]
        new[dataset]={}
        for t in record[d]:
            if t=="UPDATE_ALL":
                continue
            treatment=treatmentname[t]

            tmp = [x['x'][-1] for x in record[d][t]]
            median = np.median(tmp)
            worst = np.percentile(tmp, 100)
            w_median = 0.95 - median / datasize[d]
            w_worst = 0.95 - worst / datasize[d]
            new[dataset][treatment] = [median, worst, w_median, w_worst]

    for d in record1:
        dataset=dataname[d]
        new[dataset]={}
        for t in record1[d]:
            if t=="UPDATE_ALL":
                continue
            treatment=treatmentname[t]

            tmp = [x['x'][-1] for x in record1[d][t]]
            median=np.median(tmp)
            worst = np.percentile(tmp,100)
            w_median = 0.95 - median / datasize[d]
            w_worst = 0.95 - worst / datasize[d]
            new[dataset][treatment] = [median, worst, w_median, w_worst]

    ####draw table
    treatments = ['Auto-BM25', "Auto-Syn", "UPDATE", "REUSE", "RANDOM"]
    datasets = ['Wahono (Full)', 'Hall (Full)', "Radjenovi{\\'c} (Full)", "Kitchenham (Full)", 'Wahono (Half)',
                'Hall (Half)', "Radjenovi{\\'c} (Half)", "Kitchenham (Half)"]
    print("\\begin{tabular}{ |l|l|c|c|c|c| }")
    print("\\hline")
    print(" & & \\multicolumn{2}{|c|}{X95} & \\multicolumn{2}{|c|}{WSS@95} \\\\")
    print("\\cline{3-6}")
    print(" Dataset & Treatment & median & iqr & median & iqr \\\\")
    print("\\hline")
    for dataset in datasets:
        for treatment in treatments:
            if treatment == "UPDATE":
                d = dataset
            else:
                d = ""
            if dataset == "Kitchenham (Full)":
                if treatment == "UPDATE" or treatment == "REUSE":
                    continue
                if treatment == "Auto-Syn":
                    d = dataset
            print(d + " & " + treatment + " & " + " & ".join(
                map('{0:.0f}'.format, new[dataset][treatment][:2])) + " & " + " & ".join(
                map('{0:.2f}'.format, new[dataset][treatment][2:])) + " \\\\")
            if treatment == "RANDOM":
                print("\\hline")
                # else:
                #     print('\\cline{2-6}')
    print("\\end{tabular}")

def sum_true():
    with open("../dump/data_true.pickle","r") as handle:
        record = pickle.load(handle)
    with open("../dump/nodata1_true.pickle","r") as handle:
        record1 = pickle.load(handle)
    dataname={"Danijel.csv": "Radjenovi{\\'c} (Full)", "Wahono.csv": "Wahono (Full)", "Hall.csv": "Hall (Full)", "K_all3.csv": "Kitchenham (Full)", "Danijel2005+.csv": "Radjenovi{\\'c} (Half)", "Wahono2008+.csv": "Wahono (Half)", "Hall2007+.csv": "Hall (Half)", "K_all3+.csv": "Kitchenham (Half)"}
    datasize={"Danijel.csv": 6000, "Wahono.csv": 7002, "Hall.csv": 8991, "K_all3.csv": 1704, "Danijel2005+.csv": 3035, "Wahono2008+.csv": 3810, "Hall2007+.csv": 4066, "K_all3+.csv": 1688}
    treatmentname={"Auto_Syn": "Auto-Syn", "BM25": "Auto-BM25", "RANDOM": "RANDOM", "UPDATE_POS": "UPDATE", "REUSE": "REUSE", "Cormack_BM25": "Cormack-BM25", "Auto_Rand": "Auto-Rand"}
    new={}
    # for d in record:
    #     dataset=dataname[d]
    #     new[dataset]={}
    #     for t in record[d]:
    #         if t=="UPDATE_ALL":
    #             continue
    #         treatment=treatmentname[t]
    #
    #         tmp = [x['x'][-1] for x in record[d][t]]
    #         median=np.median(tmp)
    #         iqr = np.percentile(tmp,75)-np.percentile(tmp,25)
    #         w_median=0.95-median/datasize[d]
    #         w_iqr=(np.percentile(tmp,75)-np.percentile(tmp,25))/datasize[d]
    #
    #         new[dataset][treatment]=[median, iqr, w_median, w_iqr]

    for d in record1:
        dataset=dataname[d]
        new[dataset]={}
        for t in record1[d]:
            if t=="UPDATE_ALL":
                continue
            treatment=treatmentname[t]

            tmp = [x['x'][-1] for x in record1[d][t]]
            median=np.median(tmp)
            iqr = np.percentile(tmp,75)-np.percentile(tmp,25)
            w_median=0.95-median/datasize[d]
            w_iqr=(np.percentile(tmp,75)-np.percentile(tmp,25))/datasize[d]

            new[dataset][treatment]=[median, iqr, w_median, w_iqr]

    ####draw table
    # treatments=['Auto-BM25', "Auto-Syn", "UPDATE", "REUSE", "RANDOM"]
    treatments = ['Auto-BM25', "Auto-Syn", "RANDOM", "Cormack-BM25", "Auto-Rand"]
    # datasets = ['Wahono (Full)', 'Hall (Full)', "Radjenovi{\\'c} (Full)", "Kitchenham (Full)", 'Wahono (Half)', 'Hall (Half)', "Radjenovi{\\'c} (Half)", "Kitchenham (Half)"]
    datasets = ['Wahono (Full)', 'Hall (Full)', "Radjenovi{\\'c} (Full)", "Kitchenham (Full)"]
    print("\\begin{tabular}{ |l|l|c|c|c|c| }")
    print("\\hline")
    print(" & & \\multicolumn{2}{|c|}{X95} & \\multicolumn{2}{|c|}{WSS@95} \\\\")
    print("\\cline{3-6}")
    print(" Dataset & Treatment & median & iqr & median & iqr \\\\")
    print("\\hline")
    for dataset in datasets:
        for i,treatment in enumerate(treatments):
            if i==0:
                d=dataset
            else:
                d=""
            if dataset=="Kitchenham (Full)":
                if treatment=="UPDATE" or treatment=="REUSE":
                    continue
            print(d+" & "+treatment+ " & " + " & ".join(map('{0:.0f}'.format,new[dataset][treatment][:2])) + " & " +" & ".join(map('{0:.2f}'.format,new[dataset][treatment][2:]))+" \\\\")
            if i==4:
                print("\\hline")
            # else:
            #     print('\\cline{2-6}')
    print("\\end{tabular}")

def BM25_test():
    BM25("K_all3+.csv","software systematic review",'true')


def error_no_machine():
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    for file in files:
        print(file+": ", end='')
        BM25(file,queries[file],'est','three')

def error_machine():
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    for file in files:
        print(file+": ", end='')
        BM25(file,queries[file],'est','random')

def error_hpcc(seed = 1):
    np.random.seed(int(seed))
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    correct = ['none', 'three', 'machine']

    results={}
    for file in files:
        results[file]={}
        for cor in correct:
            print(str(seed)+": "+file+": "+ cor+ ": ", end='')
            if cor == 'three':
                result = BM25(file,queries[file],'est','three')
            elif cor == 'machine':
                result = BM25(file,queries[file],'est','random', 5)
            else:
                result = BM25(file,queries[file],'est','random')

            results[file][cor] = analyze(result)
    with open("../dump/error_hpcc.pickle","w+") as handle:
        pickle.dump(results,handle)




if __name__ == "__main__":
    eval(cmd())
