from __future__ import division, print_function



from ES_CORE import ESHandler
from model import SVM
from injest import Vessel
import numpy as np
from pdb import set_trace
from demos import cmd
from crawler import crawl_acm_doi
import pickle
import matplotlib.pyplot as plt


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
    url="http://dl.acm.org/results.cfm?query=%28software%20OR%20applicati%2A%20OR%20systems%20%29%20AND%20%28fault%2A%20OR%20defect%2A%20OR%20quality%20OR%20error-prone%29%20AND%20%28predict%2A%20OR%20prone%2A%20OR%20probability%20OR%20assess%2A%20OR%20detect%2A%20OR%20estimat%2A%20OR%20classificat%2A%29&filtered=resources%2Eft%2EresourceFormat=PDF&within=owners%2Eowner%3DHOSTED&dte=2000&bfr=2013&srt=_score"
    crawl_acm_doi(url)

def inject():
    ESHandler.injest(force=True)

def simple_exp(id):
    stepsize=10
    if container.SVM is None:
        container.also(SVM=SVM(disp=stepsize, opt=container.OPT).featurize())

    result = container.SVM.simple_active(step=stepsize, initial=10, pos_limit=1)

    with open("../dump/simple_exp" + str(id) + ".pickle","w") as f:
        pickle.dump(result,f)

    set_trace()

def simple_draw(id):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    with open("../dump/simple_exp"+str(id)+".pickle", "r") as f:
        result=pickle.load(f)
    plt.plot(result['x'], result["linear_review"], label="linear_review")
    plt.plot(result['x'], result["simple_active"], label="simple_active")
    plt.plot(result['x'], result["aggressive_undersampling"], label="aggressive_undersampling")
    plt.plot(result['x'], result["smote"], label="smote")
    plt.plot(result['x'], result["continuous_active"], label="continuous_active")
    plt.plot(result['x'], result["continuous_aggressive"], label="continuous_aggressive")
    # plt.plot(result['x'], result["semi_continuous_aggressive"], label="semi_continuous_aggressive")
    plt.plot(result['x'][result['stable']], result["simple_active"][result['stable']], color="yellow",marker='o')
    plt.plot(result['x'][result['begin']], result["simple_active"][result['begin']], color="black", marker='o')
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.35, 1), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/simple_exp" + str(id) + ".eps")
    plt.savefig("../figure/simple_exp" + str(id) + ".png")

def repeat_exp(id):
    repeats=10
    stepsize=10
    if container.SVM is None:
        container.also(SVM=SVM(disp=stepsize, opt=container.OPT).featurize())

    results=[]
    for j in xrange(repeats):
        result = container.SVM.simple_active(step=stepsize, initial=70, pos_limit=2)
        results.append(result)

    with open("../dump/repeat_exp" + str(id) + ".pickle","w") as f:
        pickle.dump(results,f)

    set_trace()

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
            result[key] = np.array(result[key])/7002
            continue
        if key == 'stable' or key == 'begin':
            continue
        result[key] = np.array(result[key]) / 62
    return result



def comp_draw(id):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    with open("../dump/simple_exp8.pickle", "r") as f:
        result8=pickle.load(f)
    with open("../dump/simple_exp10.pickle", "r") as f:
        result10 = pickle.load(f)

    ### normalize ###
    result8 = rescale(result8)
    result10 = rescale(result10)
    #################

    plt.plot(result8['x'], result8["linear_review"], label="linear_review")
    plt.plot(result8['x'], result8["aggressive_undersampling"], label="patient_aggressive_undersampling")
    plt.plot(result10['x'], result10["continuous_active"], label="hasty_continuous_active")
    plt.plot(result8['x'], result8["continuous_aggressive"], label="patient_continuous_aggressive")
    plt.plot(result10['x'], result10["aggressive_undersampling"], label="hasty_aggressive_undersampling")
    plt.plot(result10['x'], result10["continuous_aggressive"], label="hasty_continuous_aggressive")
    plt.plot(result10['x'], result10["semi_continuous_aggressive"], label="hasty_semi_continuous_aggressive")
    plt.plot(result8['x'][result8['stable']], result8["simple_active"][result8['stable']], color="yellow",marker='o')
    plt.plot(result8['x'][result8['begin']], result8["simple_active"][result8['begin']], color="white", marker='o')
    plt.plot(result10['x'][result10['stable']], result10["simple_active"][result10['stable']], color="yellow", marker='o')
    plt.plot(result10['x'][result10['begin']], result10["simple_active"][result10['begin']], color="white", marker='o')
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

    with open("../dump/repeat_exp0.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_exp1.pickle", "r") as f:
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



if __name__ == "__main__":
    eval(cmd())
