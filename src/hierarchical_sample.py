from __future__ import print_function, division
import numpy as np
from sklearn.cluster import KMeans
from my_csr import *
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

from pdb import set_trace

# __author__ = 'Zhe Yu'

def entropy(labels):
    dict=Counter(labels)
    return -sum([dict[key]/len(labels)*np.log2(dict[key]/len(labels)) for key in dict])

def first(list):
    try:
        return list[0]
    except:
        return list

class b_tree(object):
    def __init__(self,root={}):
        self.root=root
        self.left=None
        self.right=None

    def set_root(self,dic):
        self.root=dic

    def set_left(self,other):
        self.left=other

    def set_right(self,other):
        self.right=other

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_root(self):
        return self.root

    def split(self,csrmat):
        if self.root:
            axis=self.root['axis']
            splitpoint=self.root['splitpoint']
            west=[]
            east=[]
            for i,mat in enumerate(csrmat):
                if csr_dot(mat,axis)>splitpoint:
                    east.append(i)
                else:
                    west.append(i)
            return west,east
        else:
            return None,None

    def inertia(self,csrmat):
        if self.root:
            west,east=self.split(csrmat)
            return self.left.inertia(csrmat[west])+self.right.inertia(csrmat[east])
        else:
            return csr_inertia(csrmat)

"Parent"
class Hierarchical_cluster(object):
    def __init__(self,data,label,samples=10,thres=0.9):
        self.tree=b_tree()
        self.clusters=1
        self.data=data
        self.label=np.array(label)
        self.samples=samples
        self.thres=thres
        self.pool=[]

    def set_samples(self,samples):
        self.samples=samples

    def set_thres(self,thres):
        self.thres=thres

    def addlabel(self,indices):
        try:
            self.pool=list(set(self.pool)|set(indices))
        except:
            self.pool=list(set(self.pool)|set([indices]))

    def eval_inertia(self):
        return self.tree.inertia(self.data)

    def eval_intercluster(self):

        def mass(csrmat,members):
            for i in members:
                try:
                    x=x+csrmat[i]
                except:
                    x=csrmat[i]
            return x/csrmat.shape[0]

        indices,route=self.bottom()
        core=[mass(self.data,members) for members in indices]
        dist=0
        n=len(core)
        for i in xrange(n-1):
            for j in xrange(i+1,n):
                dist=dist+csr_dist(core[i],core[j])
        dist=dist/(n*(n-1)/2)
        return dist

    def bottom(self):

        def iter_bottom(part,tree,route=[]):
            if tree.root:
                tmp=[]
                routenew=[]
                west,east=tree.split(self.data[part])
                west=np.array(part[west])
                east=np.array(part[east])
                ind,routenext=iter_bottom(west,tree.left,route+[0])
                tmp.extend(ind)
                routenew.extend(routenext)
                ind,routenext=iter_bottom(east,tree.right,route+[1])
                tmp.extend(ind)
                routenew.extend(routenext)
                return tmp,routenew
            else:
                return [part],[route]

        indices,route=iter_bottom(np.array(range(self.data.shape[0])),self.tree)
        return indices,route

    "will be defined in children classes"
    def split(self,data):
        return 0,0,0

    def addcluster(self):

        indices,route=self.bottom()

        for i in xrange(len(indices)):
            already=list(set(indices[i])&set(self.pool))
            if len(already)<self.samples:
                num=self.samples-len(already)
                can=list(set(indices[i])-set(already))
                sel=list(np.random.choice(can,num,replace=False))
                self.addlabel(sel)
                already=already+sel
                ispure=Counter(self.label[already])
                if max(ispure.values())<self.thres:
                    axis,splitpoint,gain=self.split(self.data[indices[i]])
                    tree=self.tree
                    for dir in route[i]:
                        if dir==0:
                            tree=tree.left
                        else:
                            tree=tree.right
                    tree.root={'axis':axis,'splitpoint':splitpoint}
                    tree.left=b_tree()
                    tree.right=b_tree()
                    self.clusters=self.clusters+1
                    self.addcluster()
                    return self.pool
        return self.pool


    def generate_full_tree(self):
        leaf,route=self.bottom()
        lengthlist=np.argsort(map(len,route))
        lengthll=sorted(map(len,route))
        indexlist=[]
        tmp=0
        for i in xrange(1,self.clusters):
            try:
                tmp=lengthll.index(i)
            except:
                pass
            indexlist.append(tmp)
        leaf=list(np.array(leaf)[lengthlist])
        route=list(np.array(route)[lengthlist])
        result_tree={}
        while route and (not route==[[]]):
            route_a=route.pop()
            leaf_a=leaf.pop()
            result_tree[tuple(route_a)]=leaf_a
            b=[len(route)-i-1 for i in xrange(len(route)) if route_a[:-1] == route[len(route)-i-1][:-1]]
            route_b=route.pop(b[0])
            leaf_b=leaf.pop(b[0])
            result_tree[tuple(route_b)]=leaf_b
            if len(route_a)>1:
                route.insert(indexlist[len(route_a)-1],route_a[:-1])
                leaf.insert(indexlist[len(route_a)-1],list(leaf_a)+list(leaf_b))
        return result_tree


"principal component"
class Pc_cluster(Hierarchical_cluster):

    def split(self,csrmat):

        def one_dimension(csrmat,axis):
            return [[csr_dot(vec,axis)] for vec in csrmat]

        if csrmat.shape[0]<2:
            return 0,0,0
        axis=csr_pc(csrmat)
        axis=csr_l2norm(csr_matrix(axis))
        one=one_dimension(csrmat,axis)

        split=KMeans(n_clusters=2)
        split.fit(one)
        splitpoint=(split.cluster_centers_[1,0]+split.cluster_centers_[0,0])/2
        labels=split.labels_

        x1=[i for i in xrange(len(labels)) if labels[i]==0]
        x2=[i for i in xrange(len(labels)) if labels[i]==1]
        before=csr_inertia(csrmat)
        after=csr_inertia(csrmat[x1])+csr_inertia(csrmat[x2])
        gain=before-after
        return axis,splitpoint,gain



