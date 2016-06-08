from __future__ import print_function, division
from collections import Counter
from pdb import set_trace
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.feature_extraction import FeatureHasher
from sklearn import naive_bayes
from sklearn import tree
import random
from random import randint
from time import time
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import *
from my_csr import *


from ABCD import ABCD

# __author__ = 'Zhe Yu'


"Decorator to report arguments and time taken"
def run(func):
    def inner(*args, **kwargs):
        t0=time()
        print("You are running: %s" % func.__name__)
        print("Arguments were: %s, %s"%(args, kwargs))
        result = func(*args, **kwargs)
        print("Time taken: %f secs"%(time()-t0))
        return result
    return inner

def timer(func):
    def inner(*args, **kwargs):
        t0=time()
        result= func(*args,**kwargs)
        print("%s takes time: %s secs" %(func.__name__,time()-t0))
        return result
    return inner


def iqr(arr):
    return np.percentile(arr,75)-np.percentile(arr,25)


"term frequency "
def token_freqs(doc):
    return Counter(doc[1:])


"tf"
def tf(corpus):
    mat=[token_freqs(doc) for doc in corpus]
    return mat

"feature tfidf"
def tfidf_fea(corpus):
    word={}
    doc={}
    docs=0
    for row_c in corpus:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    for row in corpus:
        for key in row:
            #row[key]=(1+np.log(row[key]))*np.log(docs/doc[key])
            row[key]=row[key]*np.log(docs/doc[key])
    return corpus



"tf-idf"
def tf_idf(corpus):
    word={}
    doc={}
    docs=0
    for row_c in corpus:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    tfidf={}
    words=sum(word.values())
    for key in doc.keys():
        tfidf[key]=word[key]/words*np.log(docs/doc[key])
    return tfidf

"tf-idf_incremental"
def tf_idf_inc(row,word,doc,docs):
    docs+=1
    for key in row.keys():
        try:
            word[key]+=row[key]
        except:
            word[key]=row[key]
        try:
            doc[key]+=1
        except:
            doc[key]=1

    return word,doc,docs


"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat


"hashing trick"
def hash(mat,n_features=100, non_negative=True):
    if type(mat[0])==type('str') or type(mat[0])==type(u'unicode'):
        hasher = FeatureHasher(n_features=n_features, input_type='string', non_negative=non_negative)
    else:
        hasher = FeatureHasher(n_features=n_features, non_negative=non_negative)
    X = hasher.transform(mat)
    return X

def docfre(sub,ind,ind2):
    word={}
    doc={}
    docs=0
    for row_c in sub[ind]:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    a=doc
    word={}
    doc={}
    docs=0
    for row_c in sub[ind2]:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    b=doc
    return a,b

def ig(sub,ind,ind2):

    def nlgn(n):
        if n==0:
            return 0
        else:
            return n*np.log2(n)

    a,b=docfre(sub,ind,ind2)

    keys=list(set(a.keys()+b.keys()))

    num_a=len(ind)
    num_b=len(ind2)
    score={}
    score2={}
    for key in keys:
        if key not in a.keys():
            a[key]=0
        if key not in b.keys():
            b[key]=0
        score[key]=float(a[key])/(a[key]+b[key])
        score2[key]=float(num_a-a[key])/(num_a-a[key]+num_b-b[key])

    # score[key] is P(pos|key)
    # score2[key] is P(pos|key-)

    Ppos=num_a/(num_a+num_b)
    G={}
    for key in keys:
        G[key]=-nlgn(Ppos)+nlgn(score[key])+nlgn(score2[key])


    return G


"supervised feature selection"
def make_feature_super(dict,label,i_minor,fea='tf',norm="l2row",n_features=10000):
    sub=dict[i_minor]
    subl=label[i_minor]
    ind=[p for p in xrange(len(subl)) if subl[p]=='pos']
    ind2=list(set(range(len(subl)))-set(ind))
    score=ig(sub,ind,ind2)
    keys=np.array(score.keys())[np.argsort(score.values())[-n_features:]]
    print(len(keys))
    if fea=='tfidf_fea':
        corpus=tfidf_fea(dict)
    else:
        corpus=dict
    data=[]
    r=[]
    col=[]
    num=len(corpus)
    for i,row in enumerate(corpus):
        tmp=0
        for key in keys:
            if key in row.keys():
                data.append(row[key])
                r.append(i)
                col.append(tmp)
            tmp=tmp+1
    matt=csr_matrix((data, (r, col)), shape=(num, n_features))
    data=[]
    r=[]
    col=[]
    if norm=="l2row":
        matt=l2normalize(matt)
    elif norm=="l2col":
        matt=l2normalize(matt.transpose()).transpose()
    return matt

"make feature matrix"
def make_feature(corpus,sel="tfidf",fea='tf',norm="l2row",n_features=10000):

    if sel=="hash":
        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        matt=hash(corpus,n_features=n_features,non_negative=True)
        corpus=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()

    else:
        score={}
        if sel=="tfidf":
            score=tf_idf(corpus)
        elif sel=="docfre":
            word={}
            docs=0
            for row_c in corpus:
                word,score,docs=tf_idf_inc(row_c,word,score,docs)


        keys=np.array(score.keys())[np.argsort(score.values())][-n_features:]

        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        data=[]
        r=[]
        col=[]
        num=len(corpus)
        for i,row in enumerate(corpus):
            tmp=0
            for key in keys:
                if key in row.keys():
                    data.append(row[key])
                    r.append(i)
                    col.append(tmp)
                tmp=tmp+1
        corpus=[]
        matt=csr_matrix((data, (r, col)), shape=(num, n_features))
        data=[]
        r=[]
        col=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()
    return matt

"make feature matrix and also return vocabulary"
def make_feature_voc(corpus,sel="tfidf",fea='tf',norm="l2row",n_features=10000):

    keys=np.array([])
    if sel=="hash":
        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        matt=hash(corpus,n_features=n_features,non_negative=True)
        corpus=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()

    else:
        score={}
        if sel=="tfidf":
            score=tf_idf(corpus)
        elif sel=="docfre":
            word={}
            docs=0
            for row_c in corpus:
                word,score,docs=tf_idf_inc(row_c,word,score,docs)


        keys=np.array(score.keys())[np.argsort(score.values())][-n_features:]

        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        data=[]
        r=[]
        col=[]
        num=len(corpus)
        for i,row in enumerate(corpus):
            tmp=0
            for key in keys:
                if key in row.keys():
                    data.append(row[key])
                    r.append(i)
                    col.append(tmp)
                tmp=tmp+1
        corpus=[]
        matt=csr_matrix((data, (r, col)), shape=(num, n_features))
        data=[]
        r=[]
        col=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()


    return matt,keys



"Preprocessing: stemming + stopwords removing"
def process(txt):
  stemmer = PorterStemmer()
  cachedStopWords = stopwords.words("english")
  return ' '.join([stemmer.stem(word) for word \
                   in txt.lower().split() if word not \
                   in cachedStopWords and len(word)>1])

"resample"
def resample(data,label):
    labelCont=Counter(label)
    num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            ind=np.random.choice(id,num_s,replace=True)
            if balanced == []:
                balanced=data[np.concatenate((ind,id))]
            else:
                balanced=csr_vstack(balanced,data[np.concatenate((ind,id))])
        else:
            ind=np.random.choice(id,num,replace=False)
            if balanced == []:
                balanced=data[ind]
            else:
                balanced=csr_vstack(balanced,data[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

"half"
def half(data,label,k=5,thres=0.2):
    labelCont=Counter(label)
    if np.min(labelCont.values())>thres*np.sum(labelCont.values()):
        return data,label
    else:
        return smote_eq(data,label,k)

"smote"
def smote(data,num,k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    row=[]
    column=[]
    new=[]
    for i in xrange(num):
        mid=randint(0,data.shape[0]-1)
        nn=indices[mid,randint(1,k)]
        indx=list(set(list(data[mid].indices)+list(data[nn].indices)))
        datamade=[]
        for j in indx:
            gap=random()
            datamade.append((data[nn,j]-data[mid,j])*gap+data[mid,j])
        row.extend([i]*len(indx))
        column.extend(indx)
        new.extend(datamade)
    mat=csr_matrix((new, (row, column)), shape=(num, data.shape[1]))
    mat.eliminate_zeros()
    return mat

"smote and undersampling"
def smote_eq(data,label,k=5):
    labelCont=Counter(label)
    num=int(sum(labelCont.values())/len(labelCont.values()))
    # num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(sub)
            distances, indices = nbrs.kneighbors(sub)
            row=[]
            column=[]
            new=[]
            for i in xrange(num_s):
                mid=randint(0,sub.shape[0]-1)
                nn=indices[mid,randint(1,k)]
                indx=list(set(list(sub[mid].indices)+list(sub[nn].indices)))
                datamade=[]
                for j in indx:
                    gap=random()
                    datamade.append((sub[nn,j]-sub[mid,j])*gap+sub[mid,j])
                row.extend([i]*len(indx))
                column.extend(indx)
                new.extend(datamade)
            mat=csr_matrix((new, (row, column)), shape=(num_s, sub.shape[1]))
            if balanced == []:
                balanced=mat
            else:
                balanced=csr_vstack(balanced,mat)
            balanced=csr_vstack(balanced,sub)
        else:
            ind=np.random.choice(labelCont[l],num,replace=False)
            if balanced == []:
                balanced=sub[ind]
            else:
                balanced=csr_vstack(balanced,sub[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

"smote only oversample"
def smote_most(data,label,k=5):
    labelCont=Counter(label)
    # num=int(sum(labelCont.values())/len(labelCont.values()))
    num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(sub)
            distances, indices = nbrs.kneighbors(sub)
            row=[]
            column=[]
            new=[]
            for i in xrange(num_s):
                mid=randint(0,sub.shape[0]-1)
                nn=indices[mid,randint(1,k)]
                indx=list(set(list(sub[mid].indices)+list(sub[nn].indices)))
                datamade=[]
                for j in indx:
                    gap=random.random()
                    datamade.append((sub[nn,j]-sub[mid,j])*gap+sub[mid,j])
                row.extend([i]*len(indx))
                column.extend(indx)
                new.extend(datamade)
            mat=csr_matrix((new, (row, column)), shape=(num_s, sub.shape[1]))
            if balanced == []:
                balanced=mat
            else:
                balanced=csr_vstack(balanced,mat)
            balanced=csr_vstack(balanced,sub)
        else:
            ind=np.random.choice(labelCont[l],num,replace=False)
            if balanced == []:
                balanced=sub[ind]
            else:
                balanced=csr_vstack(balanced,sub[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

"k nearest neighbors for minority"
def neighbor(data,minor,k=5):
    def dis(first,second):
        return np.linalg.norm(first.toarray()-second.toarray(),2)

    indices=[]
    dists=[]
    pre=time()
    for ind in minor:
        print(time()-pre)
        pre=time()
        dist=[]
        indx=[]
        for i in range(data.shape[0]):
            if not dist:
                indx.append(i)
                dist.append(dis(data[ind],data[i]))
            for j,x in enumerate(dist):
                tmp=dis(data[ind],data[i])
                if tmp<x:
                    dist.insert(j,tmp)
                    indx.insert(j,i)
                    break
                if j==len(dist)-1:
                    dist.insert(j+1,tmp)
                    indx.insert(j+1,i)
            dist=dist[:k+1]
            indx=indx[:k+1]
        indices.append(indx)
        dists.append(dist)
    return indices,dists

"near smote"
def smote_near(data,label,k=5):
    labelCont=Counter(label)
    # num=int(sum(labelCont.values())/len(labelCont.values()))
    num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
    distances, indicess = nbrs.kneighbors(data)
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            # indices,dists = neighbor(data,id)
            indices=np.array(indicess)[id]

            border=[0]*len(id)
            border2=[]
            for p,x in enumerate(indices):
                for indd in x[1:]:
                    if label[indd]==l:
                        border[p]=border[p]+1
                    else:
                        break
                if border[p]>0:
                    border2.append(p)



            row=[]
            column=[]
            new=[]
            for i in xrange(num_s):
                mid=border2[randint(0,len(border2)-1)]
                nn=indices[mid,randint(1,border[mid])]
                indx=list(set(list(sub[mid].indices)+list(data[nn].indices)))
                datamade=[]
                for j in indx:
                    gap=random()
                    datamade.append((data[nn,j]-sub[mid,j])*gap+sub[mid,j])
                row.extend([i]*len(indx))
                column.extend(indx)
                new.extend(datamade)
            mat=csr_matrix((new, (row, column)), shape=(num_s, sub.shape[1]))
            if balanced == []:
                balanced=mat
            else:
                balanced=csr_vstack(balanced,mat)
            balanced=csr_vstack(balanced,sub)
        else:
            ind=np.random.choice(labelCont[l],num,replace=False)
            if balanced == []:
                balanced=sub[ind]
            else:
                balanced=csr_vstack(balanced,sub[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

"borderline smote"
def smote_border(data,label,k=5):
    labelCont=Counter(label)
    # num=int(sum(labelCont.values())/len(labelCont.values()))
    num=int(np.max(labelCont.values()))
    labelmade=[]
    balanced=[]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
    distances, indicess = nbrs.kneighbors(data)
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if labelCont[l]<num:
            num_s=num-labelCont[l]
            # indices,dists = neighbor(data,id)

            indices=np.array(indicess)[id]

            border=[[]]*len(id)
            border2=[]
            for p,x in enumerate(indices):
                for indd in x[1:]:
                    if label[indd]==l:
                        border[p]=border[p]+[indd]
                if len(border[p])>0 and len(border[p])<=int(k/2)+1:
                    border2.append(p)

            row=[]
            column=[]
            new=[]
            for i in xrange(num_s):
                mid=border2[randint(0,len(border2)-1)]
                nn=border[mid][randint(0,len(border[mid])-1)]
                indx=list(set(list(sub[mid].indices)+list(data[nn].indices)))
                datamade=[]
                for j in indx:
                    gap=random()
                    datamade.append((data[nn,j]-sub[mid,j])*gap+sub[mid,j])
                row.extend([i]*len(indx))
                column.extend(indx)
                new.extend(datamade)
            mat=csr_matrix((new, (row, column)), shape=(num_s, sub.shape[1]))
            if balanced == []:
                balanced=mat
            else:
                balanced=csr_vstack(balanced,mat)
            balanced=csr_vstack(balanced,sub)
        else:
            ind=np.random.choice(labelCont[l],num,replace=False)
            if balanced == []:
                balanced=sub[ind]
            else:
                balanced=csr_vstack(balanced,sub[ind])
    labelmade=np.array(labelmade)
    return balanced, labelmade

# "SMOTE for sparse"
# def smote_sp(data,num,k=5):
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
#     distances, indices = nbrs.kneighbors(data)
#     row=[]
#     column=[]
#     new=[]
#     for i in xrange(num):
#         mid=randint(0,data.shape[0]-1)
#         nn=indices[mid,randint(1,k)]
#         indx=list(set(list(data[mid].indices)+list(data[nn].indices)))
#         datamade=[]
#         for j in indx:
#             if (data[mid,j]==0 or data[nn,j]==0):
#                 if random()<0.5:
#                     datamade.append(0)
#                     continue
#             gap=random()
#             datamade.append((data[nn,j]-data[mid,j])*gap+data[mid,j])
#         row.extend([i]*len(indx))
#         column.extend(indx)
#         new.extend(datamade)
#     mat=csr_matrix((new, (row, column)), shape=(num, data.shape[1]))
#     mat.eliminate_zeros()
#     return mat
#
# "SMOTE for sparse alternative"
# def smote_sp1(data,num,k=5):
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(data)
#     distances, indices = nbrs.kneighbors(data)
#     row=[]
#     column=[]
#     new=[]
#     for i in xrange(num):
#         mid=randint(0,data.shape[0]-1)
#         nn=indices[mid,randint(1,k)]
#         indx=list(data[mid].indices)
#         datamade=[]
#         for j in indx:
#             gap=random()
#             datamade.append((data[nn,j]-data[mid,j])*gap+data[mid,j])
#         row.extend([i]*len(indx))
#         column.extend(indx)
#         new.extend(datamade)
#     mat=csr_matrix((new, (row, column)), shape=(num, data.shape[1]))
#     mat.eliminate_zeros()
#     return mat

"Decision Tree: CART"
def do_DT(train_data,train_label,issmote='smote',neighbors=5):
    if issmote=="smote":
        train_data,train_label=smote_most(train_data,train_label,k=neighbors)
    elif issmote=="border":
        train_data,train_label=smote_border(train_data,train_label,k=neighbors)
    elif issmote=="near":
        train_data,train_label=smote_near(train_data,train_label,k=neighbors)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_label)
    return clf

"linear"
def do_SVM(*args, **kwargs):
    return SVM('linear',*args, **kwargs)
"Polynomial"
def do_SVM_Poly(*args, **kwargs):
    return SVM('poly',*args, **kwargs)
"RBF"
def do_SVM_RBF(*args, **kwargs):
    return SVM('rbf',*args, **kwargs)
"Sigmoid"
def do_SVM_Sig(*args, **kwargs):
    return SVM('sigmoid',*args, **kwargs)


"SVM"
def SVM(kernel,train_data,train_label,issmote='smote',neighbors=5):
    if issmote=="smote":
        train_data,train_label=smote_most(train_data,train_label,k=neighbors)
    elif issmote=="border":
        train_data,train_label=smote_border(train_data,train_label,k=neighbors)
    elif issmote=="near":
        train_data,train_label=smote_near(train_data,train_label,k=neighbors)
    clf = svm.SVC(probability=True,kernel=kernel)
    clf.fit(train_data, train_label)

    return clf

def dis(first,second):
    return np.dot(first.toarray()[0],second.toarray()[0])

class KNN(object):

    def __init__(self,k=10):
        self.body=[]
        self.label=[]
        self.k=k
        self.classes_=[]

    def fit(self,data,label):
        self.body=data
        self.label=label
        self.classes_=Counter(self.label).keys()

    def __add__(self, other):
        out=KNN(k=self.k)
        if not self.body:
            out=other

        elif not other.body:
            out=self

        else:
            out.body=csr_vstack(self.body,other.body)
            out.label=np.array(list(self.label)+list(other.label))
            out.classes_=Counter(out.label).keys()
        return out



    def predict(self,tests):

        keys=[]
        for test in tests:
            dist=[]
            ll=[]
            for i,data in enumerate(self.body):
                if not dist:
                    dist.append(dis(data,test))
                    ll.append(self.label[i])
                for j,x in enumerate(dist):
                    tmp=dis(data,test)
                    if tmp>x:
                        dist.insert(j,tmp)
                        ll.insert(j,self.label[i])
                        break
                    if j==len(dist)-1:
                        dist.insert(j+1,tmp)
                        ll.insert(j+1,self.label[i])
                dist=dist[:self.k]
                ll=ll[:self.k]
            ll=Counter(ll)
            key=ll.keys()[np.argmax(ll.values())]
            keys.append(key)
        return keys

    def predict_prob(self,tests):

        out=[]
        for test in tests:
            dist=[]
            ll=[]
            for i,data in enumerate(self.body):
                if not dist:
                    dist.append(dis(data,test))
                    ll.append(self.label[i])
                for j,x in enumerate(dist):
                    tmp=dis(data,test)
                    if tmp>x:
                        dist.insert(j,tmp)
                        ll.insert(j,self.label[i])
                        break
                    if j==len(dist)-1:
                        dist.insert(j+1,tmp)
                        ll.insert(j+1,self.label[i])
                dist=dist[:self.k]
                ll=ll[:self.k]
            ll=Counter(ll)
            tmp=[ll[k]/len(self.label) for k in self.classes_]
            out.append(tmp)
        return out






"K-NN"
def do_KNN(train_data,train_label,issmote='smote',neighbors=5):
    if issmote=="smote":
        train_data,train_label=smote_most(train_data,train_label,k=neighbors)
    elif issmote=="border":
        train_data,train_label=smote_border(train_data,train_label,k=neighbors)
    elif issmote=="near":
        train_data,train_label=smote_near(train_data,train_label,k=neighbors)

    clf=KNN(k=10)
    clf.fit(train_data,train_label)
    return clf



"Naive Bayes"
def do_NB(train_data,train_label,issmote='smote',neighbors=5):
    if issmote=="smote":
        train_data,train_label=smote_most(train_data,train_label,k=neighbors)
    elif issmote=="border":
        train_data,train_label=smote_border(train_data,train_label,k=neighbors)
    elif issmote=="near":
        train_data,train_label=smote_near(train_data,train_label,k=neighbors)
    clf = naive_bayes.MultinomialNB()
    clf.fit(train_data, train_label)
    return clf





"Load data from file to list of lists"
def readfile_binary(filename='',thres=[0.02,0.07],pre='stem'):
    dict=[]
    label=[]
    targetlabel=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.lower().split(' >>> ')[0]
                label.append(doc.lower().split(' >>> ')[1].split()[0])
                if pre=='stem':
                    dict.append(Counter(process(row).split()))
                elif pre=="bigram":
                    tm=process(row).split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=process(row).split()
                    #temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    temp2=[tm[i]+' '+tm[i+1]+' '+tm[i+2] for i in xrange(len(tm)-2)]
                    dict.append(Counter(temp2+tm))
                else:
                    dict.append(Counter(row.split()))
            except:
                pass
    labellst=Counter(label)
    n=sum(labellst.values())
    while True:
        for l in labellst:
            if labellst[l]>n*thres[0] and labellst[l]<n*thres[1]:
                targetlabel=l
                break
        if targetlabel:
            break
        thres[1]=2*thres[1]
        thres[0]=0.5*thres[0]

    for i,l in enumerate(label):
        if l == targetlabel:
            label[i]='pos'
        else:
            label[i]='neg'
    label=np.array(label)
    print("Target Label: %s" %targetlabel)
    return label, dict

"Load data, multi-label"
def readfile_multilabel(filename='',pre='stem'):
    dict=[]
    label=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.lower().split(' >>> ')[0]
                label.append(doc.lower().split(' >>> ')[1].split())
                if pre=='stem':
                    dict.append(Counter(process(row).split()))
                elif pre=="bigram":
                    tm=process(row).split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=process(row).split()
                    #temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    temp2=[tm[i]+' '+tm[i+1]+' '+tm[i+2] for i in xrange(len(tm)-2)]
                    dict.append(Counter(temp2+tm))
                else:
                    dict.append(Counter(row.split()))
            except:
                pass
    label=np.array(label)
    return label, dict

"top N label is kept"
def readfile_topN(filename='',pre='stem',num=9):
    label=[]
    dict=[]
    targetlist=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.split(' >>> ')[0]
                label.append(doc.split(' >>> ')[1].split()[0])
                if pre=='stem':
                    dict.append(Counter(process(row).split()))
                elif pre=="bigram":
                    tm=process(row).split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=process(row).split()
                    #temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    temp2=[tm[i]+' '+tm[i+1]+' '+tm[i+2] for i in xrange(len(tm)-2)]
                    dict.append(Counter(temp2+tm))
                else:
                    dict.append(Counter(row.split()))
            except:
                pass
    labelcount=Counter(label)
    targetlist=np.array(labelcount.keys())[np.argsort(labelcount.values())[-num:]]

    for i,key in enumerate(label):
        if key not in targetlist:
            label[i]='others'
    label=np.array(label)
    return label,dict




##################








"Active Learning standard"
def active_learning(data,label,pool,Classify=do_SVM,issmote="no_smote",neighbors=5,step=50,last=10):
    pool=list(pool)
    x=list(len(pool)+np.array(range(last+1))*step)

    testing=list(set(range(len(label)))-set(pool))

    result={}
    result['F_M']={}
    result['F_u']={}
    result['acc']={}
    result['F_pos'] = {}
    for i in x:

        print(issmote+'_active_'+str(i))
        clf=Classify(data[pool],label[pool],issmote=issmote,neighbors=neighbors)
        prediction=clf.predict(data[testing])
        abcd=ABCD(before=label[testing],after=prediction)
        prec = abcd("Prec")
        rec = abcd("Rec")
        TP = abcd("TP")
        F_1 = abcd("F")
        acc = sum(TP.values())/len(testing)
        pop = Counter(label[testing])
        prec_M = np.mean(prec.values())
        rec_M = np.mean(rec.values())
        prec_u = sum([prec[x]*pop[x] for x in pop])/len(testing)
        rec_u = sum([rec[x]*pop[x] for x in pop])/len(testing)
        F_M = 2*prec_M*rec_M/(prec_M+rec_M)
        F_u = 2*prec_u*rec_u/(prec_u+rec_u)
        result['F_M'][i]=F_M
        result['F_u'][i]=F_u
        result['acc'][i]=acc
        result['F_pos'][i] = F_1["pos"]

        prob=clf.predict_proba(data[testing])
        certainty = [(sorted(x)[-1]-sorted(x)[-2])/(sorted(x)[-1]) for x in prob]
        uncertain=np.argsort(certainty)[:step]
        add=list(np.array(testing)[uncertain])
        pool=pool+add
        testing=list(set(testing)-set(add))

    return result

"input=[{},{},...], output={[]}"
def listin(x):

    def dictadd(a,b):
        if type(a)==type({}):
            return {x: dictadd(a[x],b[x]) for x in a}
        else:
            if type(a)!=type([]):
                a=[a]
            if type(b)!=type([]):
                b=[b]
            return a+b

    out={}
    for a in x:
        if not out:
            out=a
        else:
            out=dictadd(out,a)
    return out


