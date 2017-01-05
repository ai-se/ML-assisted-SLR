from __future__ import print_function
from __future__ import absolute_import, division
from random import randint,random
from time import time
import numpy as np
from pdb import set_trace
from demos import cmd
from models import similarity_tune



"DE, maximization"
def differential_evolution(**kwargs):

    def mutate(candidates,f,cr,xbest):
        for i in xrange(len(candidates)):
            tmp=range(len(candidates))
            tmp.remove(i)
            while True:
                abc=np.random.choice(tmp,3)
                a3=[candidates[tt] for tt in abc]
                xold=candidates[i]
                r=randint(0,xold.decnum-1)
                xnew=model(**kwargs)
                xnew.any()
                for j in xrange(xold.decnum):
                    if random()<cr or j==r:
                        xnew.dec[j]=a3[0].dec[j]+f*(a3[1].dec[j]-a3[2].dec[j])
                    else:
                        xnew.dec[j]=xold.dec[j]
                if xnew.check(): break
            if xnew.eval()>xbest.eval():
                xbest.copy(xnew)
                print("!",end="")
            elif xnew.eval()>xold.eval():
                print("+",end="")
            else:
                xnew=xold
                print(".",end="")
            yield xnew

    model=similarity_tune
    nb=10
    maxtries=10
    f=0.75
    cr=0.3
    xbest=model(**kwargs)
    candidates=[xbest]
    for i in range(1,nb):
        x=model(**kwargs)
        candidates.append(x)
        if x.eval()>xbest.eval():
            xbest.copy(x)
    for tries in range(maxtries):
        print(", Retries: %2d, : Best solution: %s, " %(tries,xbest.dec),end="")
        candidates=[xnew for xnew in mutate(candidates,f,cr,xbest)]
        print("")
    print("Best solution: %s, " %xbest.dec,"obj: %s, " %xbest.getobj(),
          "evals: %s * %s" %(nb,maxtries))
    return xbest.dec,xbest.obj


if __name__ == "__main__":
    eval(cmd())



