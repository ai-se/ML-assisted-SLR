from __future__ import print_function, division
from DE import differential_evolution
from random import shuffle
from demos import cmd
from runner import similarity_tune
import pickle
from pdb import set_trace



def exp():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_num = 10

    if rank==0:
        # file = '/Users/zhe/PycharmProjects/Datasets/StackExchange/'+str(set)+'.txt'
        the_model=similarity_tune
        dec, obj = differential_evolution(model=the_model)


	
        for i in xrange(proc_num - 1):
                comm.send("finished", dest=i + 1)

        results={"tuned_train": obj, "tuned_dec": dec}
        print(results)
        with open("../dump/tuneLDA.pickle", "w") as f:
            pickle.dump(results,f)

    else:
        while True:
            tunee = comm.recv(source=0)
            if type(tunee)==type("str"):
                break
            era = 0
            scores = []
            while True:
                i = era * proc_num + rank
                if i + 1 > tunee[-1]:
                    break
                    scores.extend(similarity_tune(tops=tunee[0], alpha=tunee[1], eta=tunee[2], seed=i))
                era = era + 1
            comm.send(scores, dest=0)











if __name__ == "__main__":
    eval(cmd())
