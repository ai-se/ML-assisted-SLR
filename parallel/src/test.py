from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt

from sk import rdivDemo

import random

from collections import Counter
import time

from mar import MAR

def test_semi(filename="Hall.csv"):
    num = 20
    read = MAR()
    read = read.create(filename)
    poses = np.where(np.array(read.body['label']) == "yes")[0]
    negs = np.where(np.array(read.body['label']) == "no")[0]
    pos_sel = np.random.choice(poses, num, replace=False)
    neg_sel = np.random.choice(negs, num*10, replace=False)

    for id in pos_sel:
        read.code_error(id)
    for id in neg_sel:
        read.code_error(id)
    read.enable_est = True
    read.get_numbers()
    a,b,c,d = read.train()
    set_trace()

def test(filename):
    p = 5

    for i in xrange(10):
        num = 10*(i+1)
        read = MAR()
        read = read.create(filename,partitions = p)
        poses = np.where(np.array(read.body['label']) == "yes")[0]
        negs = np.where(np.array(read.body['label']) == "no")[0]
        pos_sel = np.random.choice(poses, num, replace=False)
        neg_sel = np.random.choice(negs, num*10, replace=False)

        for id in pos_sel:
            read.code_error(id)
        for id in neg_sel:
            read.code_error(id)

        read.get_numbers()
        start = time.time()
        a,b,c,d = read.train_para()
        duration = time.time()-start
        print(duration)

        read.get_numbers()
        start = time.time()
        a,b,c,d = read.train()
        duration2 = time.time()-start
        print(duration2)


if __name__ == "__main__":
    eval(cmd())

