from __future__ import print_function
from __future__ import absolute_import, division
from random import uniform
from time import time
import numpy as np
from pdb import set_trace
from runner import repeat_sim


class Model(object):
    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=uniform(self.bottom[i],self.top[i])
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0]
        self.top=[0]
        self.decnum=0
        self.objnum=0
        self.dec=[]
        self.lastdec=[]
        self.obj=[]
        self.any()

    def eval(self):
        return sum(self.getobj())

    def copy(self,other):
        self.dec=other.dec[:]
        self.lastdec=other.lastdec[:]
        self.obj=other.obj[:]
        self.bottom=other.bottom[:]
        self.top=other.top[:]
        self.decnum=other.decnum
        self.objnum=other.objnum

    def getobj(self):
        return []

    def getdec(self):
        return self.dec

    def check(self):
        for i in range(0,self.decnum):
            if self.dec[i]<self.bottom[i] or self.dec[i]>self.top[i]:
                return False
        return True



class similarity_tune(Model):
    def __init__(self,filepath="",sequence=[],term=6):
        self.bottom=[0,0]
        self.top=[1,1]
        self.decnum=2
        self.objnum=1
        self.dec=[0]*self.decnum
        self.lastdec=[]
        self.obj=[]
        self.any()

    def getobj(self):

        if self.dec==self.lastdec:
            return self.obj

        self.obj = repeat_sim(alpha=self.dec[0],eta=self.dec[1])
        return self.obj




