from __future__ import division, print_function
import numpy as np
from pdb import set_trace

def delta(a,b):
    if type(a[0])==type(str('')):
        a=a[1:]
        b=b[1:]
    a=np.sort(a)
    b=np.sort(b)
    i=0
    j=0
    ag=0
    bg=0
    while(i<len(a) and j<len(b)):
        if a[i]>b[j]:
            ag=ag+len(a)-i
            j=j+1
        elif b[j]>a[i]:
            bg=bg+len(b)-j
            i=i+1
        else:
            j=j+1
    score=np.abs(ag-bg)/(len(a)*len(b))
    if score<0.147:
        effect="trivial"
    elif score<0.33:
        effect="small"
    elif score<0.474:
        effect="moderate"
    else:
        effect="large"
    return score,effect

