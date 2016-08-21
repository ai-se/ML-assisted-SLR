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
        with open("../dump/repeat_margin_" + margin + ".pickle","w") as handle:
            pickle.dump(results, handle)
    else:
        comm.send(results,dest=0)
        print("rank %d sent" %rank)


def wrap_repeat(results):
    medians={}
    iqrs={}
    medians['x'] = results[0]['x']
    iqrs['x'] = results[0]['x']
    for key in results[0].keys():
        if key == 'x' or key == 'stable' or key == 'begin':
            continue
        else:
            tmp = np.array([what[key] for what in results])
            medians[key] = np.median(tmp,axis=0)
            iqrs[key] = np.percentile(tmp,75,axis=0) - np.percentile(tmp,25,axis=0)
    return medians, iqrs

def repeat_draw(id):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_exp"+str(id)+".pickle", "r") as f:
        results=pickle.load(f)

    medians, iqrs = wrap_repeat(results)
    medians = rescale(medians)
    iqrs = rescale(iqrs)


    line, = plt.plot(medians['x'], medians["linear_review"], label="linear_review")
    plt.plot(iqrs['x'], iqrs["linear_review"], "-.", color=line.get_color())
    line, = plt.plot(medians['x'], medians["aggressive_undersampling"], label="aggressive_undersampling")
    plt.plot(iqrs['x'], iqrs["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians['x'], medians["continuous_active"], label="continuous_active")
    plt.plot(iqrs['x'], iqrs["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians['x'], medians["continuous_aggressive"], label="continuous_aggressive")
    plt.plot(iqrs['x'], iqrs["continuous_aggressive"], "-.", color=line.get_color())
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.95, 0.45), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/repeat_exp" + str(id) + ".eps")
    plt.savefig("../figure/repeat_exp" + str(id) + ".png")



def rescale(result):
    for key in result:
        if key == 'x':
            result[key] = np.array(result[key])/result[key][-1]
            continue
        if key == 'stable' or key == 'begin':
            continue
        result[key] = np.array(result[key]) / 106
    return result



def comp_draw(id):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_exp6.pickle", "r") as f:
        result1=pickle.load(f)[0]
    with open("../dump/repeat_exp5.pickle", "r") as f:
        result2 = pickle.load(f)[0]

    ### normalize ###
    result1 = rescale(result1)
    result2 = rescale(result2)
    #################

    plt.plot(result1['x'], result1["linear_review"], label="linear_review")
    plt.plot(result1['x'], result1["aggressive_undersampling"], label="patient_aggressive_undersampling")
    plt.plot(result2['x'], result2["continuous_active"], label="hasty_continuous_active")
    plt.plot(result1['x'], result1["continuous_aggressive"], label="patient_continuous_aggressive")
    plt.plot(result2['x'], result2["aggressive_undersampling"], label="hasty_aggressive_undersampling")
    plt.plot(result2['x'], result2["continuous_aggressive"], label="hasty_continuous_aggressive")
    plt.plot(result2['x'], result2["semi_continuous_aggressive"], label="hasty_semi_continuous_aggressive")
    plt.plot(result1['x'][result1['stable']], result1["simple_active"][result1['stable']], color="yellow",marker='o')
    plt.plot(result1['x'][result1['begin']], result1["simple_active"][result1['begin']], color="white", marker='o')
    plt.plot(result2['x'][result2['stable']], result2["simple_active"][result2['stable']], color="yellow", marker='o')
    plt.plot(result2['x'][result2['begin']], result2["simple_active"][result2['begin']], color="white", marker='o')
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.95, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/comp_exp" + str(id) + ".eps")
    plt.savefig("../figure/comp_exp" + str(id) + ".png")

def comp_repeat_draw(id):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_exp5.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_exp6.pickle", "r") as f:
        result1 = pickle.load(f)

    ##wrap and normalize ##
    medians0, iqrs0 = wrap_repeat(result0)
    medians0 = rescale(medians0)
    iqrs0 = rescale(iqrs0)
    medians1, iqrs1 = wrap_repeat(result1)
    medians1 = rescale(medians1)
    iqrs1 = rescale(iqrs1)
    #################

    line, = plt.plot(medians0['x'], medians0["linear_review"], label="linear_review")
    plt.plot(iqrs0['x'], iqrs0["linear_review"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="patient_aggressive_undersampling")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="hasty_continuous_active")
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["continuous_aggressive"], label="patient_continuous_aggressive")
    plt.plot(iqrs1['x'], iqrs1["continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"], label="hasty_aggressive_undersampling")
    plt.plot(iqrs0['x'], iqrs0["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_aggressive"], label="hasty_continuous_aggressive")
    plt.plot(iqrs0['x'], iqrs0["continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="hasty_semi_continuous_aggressive")
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.95, 0.50), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/comp_repeat_exp" + str(id) + ".eps")
    plt.savefig("../figure/comp_repeat_exp" + str(id) + ".png")



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
                    if abs(dist[sort_order_dist[0]]) > margin:
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
                        negs_sel = np.argsort(np.abs(train_dist))[::-1][:int(0.5*len(train6))]
                        sample6 = np.array(train6)[poses].tolist() + np.array(train6)[negs][negs_sel].tolist()
                        csr_train6, label_train6 = smote_most(csr_mat[sample6], labels[sample6])
                        clf.fit(csr_train6, label_train6)
                        pred_proba6 = clf.predict_proba(csr_mat[pool2])
                        pos_at6 = list(clf.classes_).index("yes")
                        proba6 = pred_proba6[:, pos_at6]
                        sort_order_certain6 = np.argsort(1 - proba6)
                        can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                        train6.extend(can6)
                        pos = Counter(labels[train6])["yes"]
                        pos_track6.append(pos)

                        #####################

                        pool3 = list(set(pool2) - set(can5))
                        train3 = train5[:]
                        pos_track3 = pos_track5[:]

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


                    # clf.fit(csr_mat[train3], labels[train3])
                    # pred_proba3 = clf.predict_proba(csr_mat[pool3])
                    # pos_at = list(clf.classes_).index("yes")
                    # proba3=pred_proba3[:,pos_at]
                    # sort_order_certain3 = np.argsort(1-proba3)
                    # can3 = [pool3[i] for i in sort_order_certain3[:step]]
                    # train3.extend(can3)
                    # pool3 = list(set(pool3) - set(can3))
                    # pos = Counter(labels[train3])["yes"]
                    # pos_track3.append(pos)

                    #################################

                    can2 = [pool2[i] for i in sort_order_certain2[start:start + step]]
                    train2.extend(can2)
                    pos = Counter(labels[train2])["yes"]
                    pos_track2.append(pos)


                    can5 = [pool2[i] for i in sort_order_certain5[start:start + step]]
                    train5.extend(can5)
                    pos = Counter(labels[train5])["yes"]
                    pos_track5.append(pos)

                    can6 = [pool2[i] for i in sort_order_certain6[start:start + step]]
                    train6.extend(can6)
                    pos = Counter(labels[train6])["yes"]
                    pos_track6.append(pos)

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
        result["smote"] = pos_track6
        result["continuous_aggressive"] = pos_track7

        return result






if __name__ == "__main__":
    eval(cmd())
