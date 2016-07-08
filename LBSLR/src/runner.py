from __future__ import division, print_function



from ES_CORE import ESHandler
from model import SVM
from injest import Vessel
import numpy as np
from pdb import set_trace
from demos import cmd
from crawler import crawl_acm


ESHandler = ESHandler(force_injest=False)
container = Vessel(
        OPT=None,
        SVM=None,
        round=0
)

stepsize = 50

def tag_can():
    search_string="(software OR applicati* OR systems ) AND (fault* OR defect* OR quality OR error-prone) AND (predict* OR prone* OR probability OR assess* OR detect* OR estimat* OR classificat*)"
    res=ESHandler.query_string(search_string)
    for x in res["hits"]["hits"]:
        ESHandler.set_control(x["_id"])

def tag_user():
    with open('../data/citeseerx/final_list.txt', 'rb') as f:
        target_list = f.readlines()
    for title in target_list:
        res=ESHandler.match_title(title)
        if res["hits"]["total"]:
            print(res["hits"]["hits"][0]["_source"]["title"])
            ESHandler.set_user(res["hits"]["hits"][0]["_id"])

def parse_acm():
    url="http://dl.acm.org/results.cfm?query=(software%20OR%20applicati*%20OR%20systems%20)%20AND%20(fault*%20OR%20defect*%20OR%20quality%20OR%20error-prone)%20AND%20(predict*%20OR%20prone*%20OR%20probability%20OR%20assess*%20OR%20detect*%20OR%20estimat*%20OR%20classificat*)&within=owners.owner=HOSTED&filtered=&dte=2000&bfr=2013"
    crawl_acm(url)



if __name__ == "__main__":
    eval(cmd())
