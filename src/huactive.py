from __future__ import print_function, division
from nltk.stem.porter import *
import pickle
#from mpi4py import MPI
from pdb import set_trace

import matplotlib.pyplot as plt

from sk import rdivDemo
from demos import cmd

from funcs import *
from hierarchical_sample import *

# __author__ = 'Zhe Yu'

def get_data(filename):
    filepath = 'C:/Datasets/StackExchange/SE//made/'
    # filepath='/Users/zhe/PycharmProjects/Datasets/StackExchange/SE/made/'
    # filepath='/Users/zhe/PycharmProjects/Datasets/StackExchange/'
    # filepath='/share2/zyu9/Datasets/StackExchange/'
    thres=[0.02,0.07]
    filetype=".txt"
    pre="stem"
    sel="hash"
    fea="tf"
    norm="l2row"
    n_feature=4000
    label,dict=readfile_binary(filename=filepath+filename+filetype,thres=thres,pre=pre)
    # label,dict=readfile_topN(filename=filepath+filename+filetype,pre=pre,num=20)
    dict=np.array(dict)


    dict=make_feature(dict,sel=sel,fea=fea,norm=norm,n_features=n_feature)


    return dict,label


def get_data_voc(filename):
    filepath = 'C:/Datasets/StackExchange/SE//made/'
    # filepath='/Users/zhe/PycharmProjects/Datasets/StackExchange/SE/made/'
    # filepath='/Users/zhe/PycharmProjects/Datasets/StackExchange/'
    # filepath='/share2/zyu9/Datasets/StackExchange/'
    thres = [0.02, 0.07]
    filetype = ".txt"
    pre = "stem"
    sel = "tfidf"
    fea = "tf"
    norm = "l2row"
    n_feature = 4000
    label, dict = readfile_binary(filename=filepath + filename + filetype, thres=thres, pre=pre)
    # label,dict=readfile_topN(filename=filepath+filename+filetype,pre=pre,num=20)
    dict = np.array(dict)

    dict, voc = make_feature_voc(dict, sel=sel, fea=fea, norm=norm, n_features=n_feature)

    return dict, label, voc



def draw(result,filename):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

    plt.rc('font', **font)
    paras={'lines.linewidth': 5,'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,'figure.autolayout': True,'figure.figsize': (16,8)}
    plt.rcParams.update(paras)


    # metrics=['F_M','F_u','acc']
    metrics=["F_pos"]
    k=0
    for metric in metrics:
        plt.figure(k)
        for method in result:
            x=result[method][metric]
            seq=np.argsort(x.keys())
            xx=np.array(x.keys())[seq]
            y=np.array(x.values())[seq]
            y_median=map(np.median,y)
            y_iqr=map(iqr,y)
            line,=plt.plot(xx,y_median,label="median "+method)
            plt.plot(xx,y_iqr,"-.",color=line.get_color(),label="iqr "+method)
        plt.ylabel(metric)
        plt.xlabel("Training Size")
        plt.legend(bbox_to_anchor=(0.35, 1), loc=1, ncol = 1, borderaxespad=0.)
        plt.savefig("../figure/"+filename+"_"+metric+".eps")
        plt.savefig("../figure/"+filename+"_"+metric+".png")
        k=k+1

def getpool_pc(data,label,num):
    samples=10
    cluster=Pc_cluster(data,label,samples=samples,thres=-0.05,iscertain=True)
    pos=[i for i in xrange(len(label)) if label[i]=="pos"]
    cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num/samples))
    return cluster.getpool()

def getpool_pc_sel(data,label,num):
    samples=10
    cluster=Pc_cluster_sel(data,label,samples=samples,thres=-0.05,iscertain=True)
    pos=[i for i in xrange(len(label)) if label[i]=="pos"]
    cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num/samples))
    return cluster.getpool()

def getpool_pc_unc(data,label,num):
    samples=10
    cluster=Pc_cluster(data,label,samples=samples,thres=-0.05,iscertain=False)
    pos=[i for i in xrange(len(label)) if label[i]=="pos"]
    cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num/samples))
    return cluster.getpool()

def getpool_pc_sel_unc(data,label,num):
    samples=10
    cluster=Pc_cluster_sel(data,label,samples=samples,thres=-0.05,iscertain=False)
    pos=[i for i in xrange(len(label)) if label[i]=="pos"]
    cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num/samples))
    return cluster.getpool()

def getpool_dt(data,label,num):
    samples = 10
    cluster = Dt_cluster(data, label, samples=samples, thres=-0.05, iscertain=False)
    # pos = [i for i in xrange(len(label)) if label[i] == "pos"]
    # cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num / samples))
    return cluster.getpool()

def getpool_dt_unc(data,label,num):
    samples = 10
    cluster = Dt_cluster(data, label, samples=samples, thres=-0.05, iscertain=True)
    # pos = [i for i in xrange(len(label)) if label[i] == "pos"]
    # cluster.addlabel(random.choice(pos))
    cluster.addclusters(int(num / samples))
    return cluster.getpool()

def getpool_kmeans(data,label,num):
    samples=1
    cluster=KMeans(n_clusters=int(num/samples))
    cluster.fit(data)
    result=cluster.labels_
    x=list(set(result))
    pool=[]
    for key in x:
        a=[i for i in xrange(len(label))if result[i]==key]
        pool.extend(list(np.random.choice(a,samples,replace=False)))
    return pool

def getpool_kmeans_iter(data,label,num):
    init=int(num/2)
    cluster=KMeans(n_clusters=init)
    cluster.fit(data)
    result=cluster.labels_
    x=list(set(result))
    pool=[]
    a=[]
    whole=[]
    for key in x:
        tmp=[i for i in xrange(len(label))if result[i]==key]
        pick=list(np.random.choice(tmp,1,replace=False))
        tmp.remove(pick[0])
        a.append(tmp)
        pool.append(pick)
        whole.extend(pick)
    for iter in xrange(num-init):
        Q=Counter(whole)
        N=sum(Q.values())
        risk=[]
        for i in range(len(x)):
            P=Counter(pool[i])
            M=sum(P.values())
            pro=0
            for key in P.keys():
                pro+=np.log2(Q[key]/N)*P[key]/M
            risk.append(-pro)
        sel=random_pro(risk)
        while(not a[sel]):
            sel=random_pro(risk)
        pick=np.random.choice(a[sel],1,replace=False)[0]
        a[sel].remove(pick)
        pool[sel].append(pick)
        whole.append(pick)

    return whole

def getpool_random(data,label,num):
    x=range(len(label))
    pool=np.random.choice(x,num,replace=False)
    return pool




def test(filename):
    methods=[getpool_dt,getpool_dt_unc,getpool_random,getpool_kmeans_iter,getpool_kmeans]

    data,label=get_data(filename)
    print(Counter(label))

    repeats=25


    result=[]
    entr=[]
    for i in xrange(repeats):
        result_tmp={}
        entr_tmp={}
        for method in methods:
            pool=getpool_random(data,label,500)
            entr_tmp[method.__name__ ]=entropy(label[pool])
            result_tmp[method.__name__ ]=active_learning(data,label,pool,issmote="no_smote",step=50,last=10)
        result.append(result_tmp)
        entr.append(entr_tmp)
    entr=listin(entr)
    result=listin(result)
    with open('../dump/init_dt_'+filename+'.pickle', 'wb') as handle:
        pickle.dump(entr, handle)
    with open('../dump/result_dt_'+filename+'.pickle', 'wb') as handle:
        pickle.dump(result, handle)
    set_trace()



def show(filename):
    # methods=[getpool_random,getpool_kmeans_iter,getpool_kmeans,getpool_pc,getpool_pc_sel,getpool_pc_unc,getpool_pc_sel_unc]
    methods = [getpool_dt, getpool_dt_unc, getpool_random, getpool_kmeans_iter, getpool_kmeans]

    with open('../dump/init_dt_'+filename+'.pickle', 'rb') as handle:
        entr=pickle.load(handle)
    with open('../dump/result_dt_'+filename+'.pickle', 'rb') as handle:
        result=pickle.load(handle)

    for method in methods:
        print(method.__name__ +":")
        print(np.median(entr[method.__name__ ]))
        print(iqr(entr[method.__name__ ]))

    draw(result,filename)

def options(filename):
    data, label, voc = get_data_voc(filename)
    std=csr_stds(data)
    voc_order=voc[np.argsort(-std)]
    set_trace()
    print()


if __name__ == '__main__':
    eval(cmd())