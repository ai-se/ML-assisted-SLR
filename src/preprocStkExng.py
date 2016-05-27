from __future__ import print_function, division
__author__ = 'Zhe Yu'
from os import walk
from pdb import set_trace
import numpy as np
import random

def random_pro(risk):
    N=sum(risk)
    p=random.random()*N
    for i in range(len(risk)):
        if sum(risk[:i+1])>p:
            return i

def name(file):
  return file[:-4]

def explore(dir='/Users/zhe/PycharmProjects/Datasets/StackExchange/SE/'):
  risk=[0.01,0.01,0.1,0.3,0.58]
  num=len(risk)
  for (dirpath, dirnames, filenames) in walk(dir): break
  filenames=[f for f in filenames if f[-3:]=='txt']
  for i in xrange(10):
    random.shuffle(filenames)
    sample=filenames[:num]
    body={}
    for file in sample:
      b=[]
      with open(dir+file) as fp:
        for n, line in enumerate(fp):
          b.append(line.split(' >>> ')[0])
      print(file)
      print(len(b))
      tmp=range(len(b))
      random.shuffle(tmp)
      b=list(np.array(b)[tmp])
      body.update({file:b})

    tot=5000*i
    x=""
    for j in xrange(tot):
      which=random_pro(risk)
      x = x + body[sample[which]].pop() + " >>> "+name(sample[which])+"\n"

    with open('/Users/zhe/PycharmProjects/Datasets/StackExchange/SE/made/SE%d.txt'%(int(i)), 'w') as writer:
      writer.write(x)



  # ----- Debug -----

if __name__=="__main__":
  explore()