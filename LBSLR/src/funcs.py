from __future__ import print_function, division
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np
import random
from pdb import set_trace
from time import time


"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat

"csr vector dot product"
def csr_dot(a,b):
    x=0
    ind_a=a.indices
    ind_b=b.indices
    for ind in ind_a:
        if ind in ind_b:
            x=x+a[0,ind]*b[0,ind]
    return x

"Concatenate two csr into one (equal num of columns)"
def csr_vstack(a,b):
    if not csr_check(a):
        return b
    if not csr_check(b):
        return a
    data=list(a.data)+list(b.data)
    ind=list(a.indices)+list(b.indices)
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

