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
    filetype=".txt"
    thres=[0.3,0.7]
    pre="stem"
    sel="hash"
    fea="tf"
    norm="l2row"
    n_feature=4000
    label,dict=readfile(filename=filepath+filename+filetype,thres=thres,pre=pre)

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



def test(filename):
    step=50
    data,label=get_data(filename)
    print(Counter(label))
    cluster=Pc_cluster(data,label,samples=10,thres=0.95)
    F={}
    for i in xrange(10):
        pool=cluster.addcluster()
        x=len(pool)
        print(x)
        can=list(set(range(len(label)))-set(pool))
        issmote="no_smote"
        clf=do_SVM(data[pool],label[pool],issmote=issmote)
        predictions=clf.predict(data[can])

        abcd=ABCD(before=label[can],after=predictions)
        ll=list(set(label[can]))
        tmp = np.array([k.stats()[-2] for k in abcd()])
        if ll[0]=='pos':
            F[x]=tmp[0]
        else:
            F[x]=tmp[1]

        if clf.classes_[0]=='pos':
            ind=0
        else:
            ind=1
        prob=np.array(clf.predict_proba(data[test])[:,ind])
        proba=np.abs(0.5-prob)
        proba=np.argsort(proba)[:step]
        add=np.array(can)[proba]
        cluster.addlabel(list(add))

    draw(F)




if __name__ == '__main__':
    eval(cmd())