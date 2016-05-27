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
    filepath='/Users/zhe/PycharmProjects/Datasets/StackExchange/'
    #filepath='/share2/zyu9/Datasets/StackExchange/'
    thres=[0.01,0.05]
    filetype=".txt"
    pre="stem"
    sel="hash"
    fea="tf"
    norm="l2row"
    n_feature=4000
    # label,dict=readfile_binary(filename=filepath+filename+filetype,thres=thres,pre=pre)
    label,dict=readfile_topN(filename=filepath+filename+filetype,pre=pre,num=20)
    dict=np.array(dict)

    if not sel=="supervised":
        dict=make_feature(dict,sel=sel,fea=fea,norm=norm,n_features=n_feature)


    return dict,label



def draw(F):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}

    plt.rc('font', **font)
    paras={'lines.linewidth': 5,'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,'figure.autolayout': True,'figure.figsize': (16,8)}
    plt.rcParams.update(paras)

    seq=np.argsort(F.keys())
    x=F.keys()[seq]
    y=F.values()[seq]
    plt.plot(x,y)
    plt.savefig("../figure/test.png")

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
    methods=[getpool_random,getpool_kmeans_iter,getpool_kmeans,getpool_pc,getpool_pc_sel,getpool_pc_unc,getpool_pc_sel_unc]

    data,label=get_data(filename)
    print(Counter(label))

    repeats=25


    result={}
    for method in methods:
        result[method.__name__ ]=[]
        for i in xrange(repeats):
            pool1=getpool_random(data,label,100)
            result[method.__name__ ].append(entropy(label[pool1]))
            print("%s: %d" %(method.__name__ ,i))
        print(method.__name__)
        print(Counter(label[pool1]))

    with open('../dump/init_'+filename+'.pickle', 'wb') as handle:
        pickle.dump(result, handle)

    for method in methods:
        print(method.__name__ +":")
        print(np.median(result[method.__name__ ]))
        print(iqr(result[method.__name__ ]))



def show(filename):
    methods=[getpool_random,getpool_kmeans_iter,getpool_kmeans,getpool_pc,getpool_pc_sel,getpool_pc_unc,getpool_pc_sel_unc]

    with open('../dump/init_'+filename+'.pickle', 'rb') as handle:
        result=pickle.load(handle)

    for method in methods:
        print(method.__name__ +":")
        print(np.median(result[method.__name__ ]))
        print(iqr(result[method.__name__ ]))

if __name__ == '__main__':
    eval(cmd())