from __future__ import division, print_function




import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm


from funcs import *

from mpi4py import MPI


def simple_exp(margin):
    repeats=10
    stepsize=10
    set="Hall"
    with open("../dump/"+set+".pickle","rb") as handle:
    # with open("/share2/zyu9/Datasets/SLR/dump/"+set+".pickle","r") as handle:
        csr_mat = pickle.load(handle)
        labels = pickle.load(handle)

    result = simple_active_hpc(csr_mat,labels,step=stepsize, initial=500, pos_limit=2, margin=float(margin))




def repeat_exp(margin):
    repeats=10
    stepsize=10
    set="Hall"
    # with open("../dump/"+set+".pickle","rb") as handle:
    with open("/share2/zyu9/Datasets/SLR/dump/"+set+".pickle","rb") as handle:
        csr_mat = pickle.load(handle)
        labels = pickle.load(handle)



    results=[]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("rank: %d" %rank)
    proc_num = 5
    era=0
    while True:
        k=era*proc_num+rank
        if k+1 > repeats:
            break
        result = simple_active_hpc(csr_mat,labels,step=stepsize, initial=10, pos_limit=1, margin=float(margin))
        print("repeat: %d" %k)

        results.append(result)
        era+=1

    if rank == 0:
        for i in range(proc_num-1):
            tmp=comm.recv(source=i+1)
            results.extend(tmp)
            print("rand %d received" %i)
        with open("../dump/repeat_margin_" + str(margin) + ".pickle","w") as handle:
            pickle.dump(results, handle)
    else:
        comm.send(results,dest=0)
        print("rank %d sent" %rank)


def repeat_Hall(pos_limit):
    repeats=10
    stepsize=10
    set="Hall"
    # with open("../dump/"+set+".pickle","rb") as handle:
    with open("/share2/zyu9/Datasets/SLR/dump/"+set+".pickle","rb") as handle:
        csr_mat = pickle.load(handle)
        labels = pickle.load(handle)



    results=[]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("rank: %d" %rank)
    proc_num = 5
    era=0
    while True:
        k=era*proc_num+rank
        if k+1 > repeats:
            break
        result = simple_active_hpc(csr_mat,labels,step=stepsize, initial=10, pos_limit=int(pos_limit),margin=0.7)
        print("repeat: %d" %k)

        results.append(result)
        era+=1

    if rank == 0:
        for i in range(proc_num-1):
            tmp=comm.recv(source=i+1)
            results.extend(tmp)
            print("rand %d received" %i)
        with open("../dump/repeat_Hall_" + str(pos_limit) + ".pickle","w") as handle:
            pickle.dump(results, handle)
    else:
        comm.send(results,dest=0)
        print("rank %d sent" %rank)

def repeat_ieee(pos_limit):
    repeats=10
    stepsize=10
    set="ieee"
    # with open("../dump/"+set+".pickle","rb") as handle:
    with open("/share2/zyu9/Datasets/SLR/dump/"+set+".pickle","rb") as handle:
        csr_mat = pickle.load(handle)
        labels = pickle.load(handle)



    results=[]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("rank: %d" %rank)
    proc_num = 5
    era=0
    while True:
        k=era*proc_num+rank
        if k+1 > repeats:
            break
        result = simple_active_hpc(csr_mat,labels,step=stepsize, initial=10, pos_limit=int(pos_limit),margin=1)
        print("repeat: %d" %k)

        results.append(result)
        era+=1

    if rank == 0:
        for i in range(proc_num-1):
            tmp=comm.recv(source=i+1)
            results.extend(tmp)
            print("rand %d received" %i)
        with open("../dump/repeat_ieee_" + str(pos_limit) + ".pickle","w") as handle:
            pickle.dump(results, handle)
    else:
        comm.send(results,dest=0)
        print("rank %d sent" %rank)


def simple_active_hpc(csr_mat, labels, step=10 ,initial=200, pos_limit=5, margin=1):

        num=len(labels)
        pool=range(num)
        train=[]
        steps = np.array(range(int(num / step))) * step

        pos=0
        pos_track=[0]
        is_stable=False
        clf = svm.SVC(kernel='linear', probability=True)
        start=0
        stable=0
        begin=0
        result={}
        enough=False
        for idx, round in enumerate(steps[:-1]):
            can = np.random.choice(pool, step, replace=False)
            train.extend(can)
            pool = list(set(pool) - set(can))
            try:
                pos = Counter(labels[train])["yes"]
            except:
                pos = 0
            pos_track.append(pos)

            if not begin:
                pool2=pool[:]
                train2=train[:]
                pos_track2=pos_track[:]
                pool3 = pool2[:]
                train3 = train2[:]
                pos_track3 = pos_track2[:]
                pool4 = pool2[:]
                train4 = train2[:]
                pos_track4 = pos_track2[:]
                pool7 = pool2[:]
                train7 = train2[:]
                pos_track7 = pos_track2[:]
                if round >= initial and pos>=pos_limit:
                    begin=idx+1
            else:
                clf.fit(csr_mat[train4], labels[train4])
                pred_proba4 = clf.predict_proba(csr_mat[pool4])
                pos_at = list(clf.classes_).index("yes")
                proba4 = pred_proba4[:, pos_at]
                sort_order_certain4 = np.argsort(1 - proba4)
                can4 = [pool4[i] for i in sort_order_certain4[:step]]
                train4.extend(can4)
                pool4 = list(set(pool4) - set(can4))
                pos = Counter(labels[train4])["yes"]
                pos_track4.append(pos)


                ################ new *_C_C_A
                if not enough:
                    if pos>=10:
                        enough=True
                        pos_track9=pos_track4[:]
                        train9=train4[:]
                        pool9=pool4[:]
                else:
                    clf.fit(csr_mat[train9], labels[train9])
                    poses = np.where(labels[train9] == "yes")[0]
                    negs = np.where(labels[train9] == "no")[0]
                    train_dist = clf.decision_function(csr_mat[train9][negs])
                    negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                    sample9 = np.array(train9)[poses].tolist() + np.array(train9)[negs][negs_sel].tolist()

                    clf.fit(csr_mat[sample9], labels[sample9])
                    pred_proba9 = clf.predict_proba(csr_mat[pool9])
                    pos_at = list(clf.classes_).index("yes")
                    proba9 = pred_proba9[:, pos_at]
                    sort_order_certain9 = np.argsort(1 - proba9)
                    can9 = [pool9[i] for i in sort_order_certain9[:step]]
                    train9.extend(can9)
                    pool9 = list(set(pool9) - set(can9))
                    pos = Counter(labels[train9])["yes"]
                    pos_track9.append(pos)

                ############################

                ### continuous aggressive
                clf.fit(csr_mat[train7], labels[train7])
                poses = np.where(labels[train7] == "yes")[0]
                negs = np.where(labels[train7] == "no")[0]
                train_dist = clf.decision_function(csr_mat[train7][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample7 = np.array(train7)[poses].tolist() + np.array(train7)[negs][negs_sel].tolist()

                clf.fit(csr_mat[sample7], labels[sample7])
                pred_proba7 = clf.predict_proba(csr_mat[pool7])
                pos_at = list(clf.classes_).index("yes")
                proba7 = pred_proba7[:, pos_at]
                sort_order_certain7 = np.argsort(1 - proba7)
                can7 = [pool7[i] for i in sort_order_certain7[:step]]
                train7.extend(can7)
                pool7 = list(set(pool7) - set(can7))
                pos = Counter(labels[train7])["yes"]
                pos_track7.append(pos)





                if not is_stable:
                    clf.fit(csr_mat[train2], labels[train2])
                    pred_proba = clf.predict_proba(csr_mat[pool2])
                    # sort_order_uncertain = np.argsort(np.abs(pred_proba[:,0] - 0.5))
                    dist = clf.decision_function(csr_mat[pool2])
                    sort_order_dist = np.argsort(np.abs(dist))
                    if abs(dist[sort_order_dist[0]]) > margin or round == steps[-2]:
                        is_stable = True
                        stable=idx


                        train5 = train2[:]
                        pos_track5 = pos_track2[:]
                        train6 = train2[:]
                        pos_track6 = pos_track2[:]


                        pos_at = list(clf.classes_).index("yes")
                        proba = pred_proba[:, pos_at]
                        sort_order_certain2 = np.argsort(1 - proba)
                        can2 = [pool2[i] for i in sort_order_certain2[start:start + step]]

                        train2.extend(can2)
                        pos = Counter(labels[train2])["yes"]
                        pos_track2.append(pos)


                        ### data balancing ###
                        ### Agressive undersampling ####
                        poses=np.where(labels[train5] == "yes")[0]
                        negs=np.where(labels[train5] == "no")[0]
                        train_dist = clf.decision_function(csr_mat[train5][negs])
                        negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                        sample5 = np.array(train5)[poses].tolist()+np.array(train5)[negs][negs_sel].tolist()
                        clf.fit(csr_mat[sample5], labels[sample5])
                        pred_proba5 = clf.predict_proba(csr_mat[pool2])
                        pos_at5 = list(clf.classes_).index("yes")
                        proba5 = pred_proba5[:, pos_at5]
                        sort_order_certain5 = np.argsort(1 - proba5)
                        can5 = [pool2[i] for i in sort_order_certain5[start:start + step]]
                        train5.extend(can5)
                        pos = Counter(labels[train5])["yes"]
                        pos_track5.append(pos)

                        ### SMOTE ####
                        # negs_sel = np.argsort(np.abs(train_dist))[::-1][:int(0.5*len(train6))]
                        # sample6 = np.array(train6)[poses].tolist() + np.array(train6)[negs][negs_sel].tolist()
                        # csr_train6, label_train6 = smote_most(csr_mat[sample6], labels[sample6])
                        # clf.fit(csr_train6, label_train6)
                        # pred_proba6 = clf.predict_proba(csr_mat[pool2])
                        # pos_at6 = list(clf.classes_).index("yes")
                        # proba6 = pred_proba6[:, pos_at6]
                        # sort_order_certain6 = np.argsort(1 - proba6)
                        # can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                        # train6.extend(can6)
                        # pos = Counter(labels[train6])["yes"]
                        # pos_track6.append(pos)

                        #####################

                        pool3 = list(set(pool2) - set(can5))
                        train3 = train5[:]
                        pos_track3 = pos_track5[:]

                        pool8 = pool3[:]
                        train8 = train3[:]
                        pos_track8 = pos_track3[:]

                        start = start + step
                    else:
                        # can2 = [pool2[i] for i in sort_order_uncertain[:step]]
                        can2 = [pool2[i] for i in sort_order_dist[:step]]
                        train2.extend(can2)
                        pool2 = list(set(pool2) - set(can2))
                        pos = Counter(labels[train2])["yes"]
                        pos_track2.append(pos)

                else:
                    #### semi_continuous_aggressive
                    clf.fit(csr_mat[train3], labels[train3])
                    poses = np.where(labels[train3] == "yes")[0]
                    negs = np.where(labels[train3] == "no")[0]
                    train_dist = clf.decision_function(csr_mat[train3][negs])
                    negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                    sample3 = np.array(train3)[poses].tolist() + np.array(train3)[negs][negs_sel].tolist()

                    clf.fit(csr_mat[sample3], labels[sample3])
                    pred_proba3 = clf.predict_proba(csr_mat[pool3])
                    pos_at = list(clf.classes_).index("yes")
                    proba3 = pred_proba3[:, pos_at]
                    sort_order_certain3 = np.argsort(1 - proba3)
                    can3 = [pool3[i] for i in sort_order_certain3[:step]]
                    train3.extend(can3)
                    pool3 = list(set(pool3) - set(can3))
                    pos = Counter(labels[train3])["yes"]
                    pos_track3.append(pos)

                    #### semi_continuous
                    clf.fit(csr_mat[train8], labels[train8])
                    pred_proba8 = clf.predict_proba(csr_mat[pool8])
                    pos_at = list(clf.classes_).index("yes")
                    proba8=pred_proba8[:,pos_at]
                    sort_order_certain8 = np.argsort(1-proba8)
                    can8 = [pool8[i] for i in sort_order_certain8[:step]]
                    train8.extend(can8)
                    pool8 = list(set(pool8) - set(can8))
                    pos = Counter(labels[train8])["yes"]
                    pos_track8.append(pos)

                    #################################

                    can2 = [pool2[i] for i in sort_order_certain2[start:start + step]]
                    train2.extend(can2)
                    pos = Counter(labels[train2])["yes"]
                    pos_track2.append(pos)


                    can5 = [pool2[i] for i in sort_order_certain5[start:start + step]]
                    train5.extend(can5)
                    pos = Counter(labels[train5])["yes"]
                    pos_track5.append(pos)

                    # can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                    # train6.extend(can6)
                    # pos = Counter(labels[train6])["yes"]
                    # pos_track6.append(pos)

                    start = start + step


            print("Round #{id} passed\r".format(id=round), end="")

        result["begin"] = begin
        result["stable"] = stable
        result["x"] = steps
        result["linear_review"] = pos_track
        result["simple_active"] = pos_track2
        result["semi_continuous_aggressive"] = pos_track3
        result["continuous_active"] = pos_track4
        result["aggressive_undersampling"] = pos_track5
        # result["smote"] = pos_track6
        result["continuous_aggressive"] = pos_track7
        result["semi_contunuous"] = pos_track8

        result["new_continuous_aggressive"] = pos_track9

        return result





if __name__ == "__main__":
    eval(cmd())
