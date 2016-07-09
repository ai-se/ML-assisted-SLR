from __future__ import print_function, division
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np
import random
from pdb import set_trace
from time import time

# __author__ = 'Zhe Yu'


"Concatenate two csr into one (equal num of columns)"
def csr_vstack(a,b):
    data=np.array(list(a.data)+list(b.data))
    ind=np.array(list(a.indices)+list(b.indices))
    indp=list(a.indptr)+list(b.indptr+a.indptr[-1])[1:]
    return csr_matrix((data,ind,indp),shape=(a.shape[0]+b.shape[0],a.shape[1]))

"check is csr vector"
def csr_checkvec(a):
    if not csr_check(a):
        return False
    elif not a.shape[0]==1:
        print("not a vector")
        return False
    else:
        return True

"check is csr"
def csr_check(a):
    if not type(a)==type(csr_matrix([1,2])):
        return False
    else:
        return True

"csr vector dot product"
def csr_dot(a,b):
    if csr_checkvec(a) and csr_checkvec(b):
        x=0
        ind_a=a.indices
        ind_b=b.indices
        for ind in ind_a:
            if ind in ind_b:
                x=x+a[0,ind]*b[0,ind]
        return x

"csr vector euclidean distance"
def csr_dist(a,b):
    if csr_checkvec(a) and csr_checkvec(b):
        return np.linalg.norm((a-b).data)

"find max distance row from row i"
def csr_maxdist(csrmat,i):
    if csr_check(csrmat):
        n=csrmat.shape[0]
        max_dist=0
        max_ind=i
        for j in xrange(n):
            dist=csr_dist(csrmat[i],csrmat[j])
            if dist>max_dist:
                max_dist=dist
                max_ind=j
        return max_ind

"find a pair of poles"
def csr_poles(csrmat):
    if csr_check(csrmat):
        n=csrmat.shape[0]
        first=random.randint(0,n-1)
        second=csr_maxdist(csrmat,first)
        third=csr_maxdist(csrmat,second)
        return second,third

"Fast principal component"
def csr_pc(csrmat):
    ux,sx,vx=linalg.svds(csrmat,k=1)
    return vx[-1]

"normalize to unit 1 length"
def csr_l2norm(csrmat):
    if csr_check(csrmat):
        csrmat=csrmat.asfptype()
        for i in xrange(csrmat.shape[0]):
            nor=np.linalg.norm(csrmat[i].data)
            if (not nor==0) and (not nor==1):
                for k in csrmat[i].indices:
                    csrmat[i,k]=csrmat[i,k]/nor
        return csrmat

"sum distance to the center of mass"
def csr_inertia(csrmat):
    if csr_check(csrmat):
        center=csr_matrix([0]*csrmat.shape[1])
        for mat in csrmat:
            center=center+mat
        center=center/csrmat.shape[0]
        sum=0
        for mat in csrmat:
            sum=sum+csr_dist(mat,center)
        return sum

"iqr"
def csr_iqr(csrvec):
    return np.percentile(csrvec.toarray(),75)-np.percentile(csrvec.toarray(),25)

"projection"
def one_dimension(csrmat, axis):
    return csrmat * axis.transpose()

"variance"
def csr_std(csrvec):
    xbar=csrvec.mean()
    values=csrvec.data
    n=csrvec.shape[1]
    if n<2:
        return 0
    y=(n-len(values))*(xbar**2)
    for x in values:
        y=y+(x-xbar)**2
    return np.sqrt(y/(n-1))

def csr_stds(csrmat,axis=1):
    if axis==1:
        tmp=csrmat.transpose()
    else:
        tmp=csrmat
    return np.array([csr_std(x) for x in tmp])

"diameter"
def csr_diameters(csrmat):
    return (csrmat.max(0)-csrmat.min(0)).toarray()[0]