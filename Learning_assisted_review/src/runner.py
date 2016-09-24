from __future__ import division, print_function



from ES_CORE import ESHandler
from model import SVM
from injest import Vessel,defaults
import numpy as np
from pdb import set_trace
from demos import cmd
from crawler import crawl_acm_doi
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sk import rdivDemo
import unicodedata
from sklearn import svm
from collections import Counter
from scipy.sparse import csr_matrix


es = ESHandler(force_injest=False)
container = Vessel(
        OPT=None,
        SVM=None,
        round=0
)

stepsize = 50





def colorcode(N):
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=N-1, clip=True)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap

def saveData(set):
    stepsize = 10

    if container.SVM is None:
        container.also(SVM=SVM(disp=stepsize, set=set, opt=container.OPT).featurize())
    container.SVM.saveData()

def splitData(set,year):
    stepsize = 10

    if container.SVM is None:
        container.also(SVM=SVM(disp=stepsize, set=set, opt=container.OPT).featurize())
    container.SVM.splitData(year)


def export_CSV(set):
    es=ESHandler(es=defaults(TYPE_NAME=set),force_injest=False)
    res=es.get_unlabeled()
    csv_content=u'id,title,abstract,label\n'
    for x in res['hits']['hits']:
        csv_content=csv_content+x['_id']+','+x['_source']['title']+","+x['_source']['abstract']+","+x['_source']['user']+"\n"
    with open("../dump/" + str(set) + ".csv","w") as f:
        f.write(unicodedata.normalize('NFKD', csv_content).encode('ascii', 'ignore'))

def tag_can():
    search_string="(software OR applicati* OR systems ) AND (fault* OR defect* OR quality OR error-prone) AND (predict* OR prone* OR probability OR assess* OR detect* OR estimat* OR classificat*)"
    res=es.query_string(search_string)
    for x in res["hits"]["hits"]:
        es.set_control(x["_id"])

def tag_user():
    with open('../data/citeseerx/final_list.txt', 'rb') as f:
        target_list = f.readlines()
    for title in target_list:
        res=es.match_title(title)
        if res["hits"]["total"]:
            print(res["hits"]["hits"][0]["_source"]["title"])
            es.set_user(res["hits"]["hits"][0]["_id"])

def parse_acm():
    url="http://dl.acm.org/results.cfm?query=%28software%20OR%20applicati%2A%20OR%20systems%20%29%20AND%20%28fault%2A%20OR%20defect%2A%20OR%20quality%20OR%20error-prone%29%20AND%20%28predict%2A%20OR%20prone%2A%20OR%20probability%20OR%20assess%2A%20OR%20detect%2A%20OR%20estimat%2A%20OR%20classificat%2A%29&filtered=resources%2Eft%2EresourceFormat=PDF&within=owners%2Eowner%3DHOSTED&dte=2000&bfr=2013&srt=_score"
    crawl_acm_doi(url)

def injest():
    es.injest(force=True)

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
        container.also(SVM=SVM(disp=stepsize, set="Hall", opt=container.OPT).featurize())

    results=[]
    for j in xrange(repeats):
        result = container.SVM.simple_active(step=stepsize, initial=10, pos_limit=5)
        results.append(result)

    with open("../dump/repeat_exp" + str(id) + ".pickle","w") as f:
        pickle.dump(results,f)

    set_trace()



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


def wrap_repeat(results):
    medians={}
    iqrs={}
    medians['x'] = results[0]['x']
    iqrs['x'] = results[0]['x']
    for key in results[0].keys():
        if key == 'x':
            continue
        elif key == 'stable':
            tmp = np.array([what[key] for what in results])
            medians[key] = np.percentile(tmp,28)
            iqrs[key] = np.percentile(tmp,75) - np.percentile(tmp,25)
        elif key == 'begin':
            tmp = np.array([what[key] for what in results])
            medians[key] = np.percentile(tmp,48)
            iqrs[key] = np.percentile(tmp,75) - np.percentile(tmp,25)
        else:
            tmp = np.array([what[key] for what in results])
            medians[key] = np.median(tmp,axis=0)
            iqrs[key] = np.percentile(tmp,75,axis=0) - np.percentile(tmp,25,axis=0)
    return medians, iqrs


def rescale(result):
    for key in result:
        if key == 'x':
            result[key] = np.array(result[key])/result[key][-1]
            continue
        if key == 'stable' or key == 'begin':
            continue
        result[key] = np.array(result[key]) / 106
        # result[key] = np.array(result[key]) / 62
    return result

def rescaleY(result,doc):
    for key in result:
        if key == 'x':
            continue
        if key == 'stable' or key == 'begin':
            continue
        result[key] = np.array(result[key]) / doc
    return result

def cutListinDict(dict, Display):
    return {key:dict[key][:Display] if (key!="begin" and key!="stable") else dict[key] for key in dict.keys()}



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

    N= 10

    indices = range(N)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    scalarMap = colorcode(N)

    with open("../dump/repeat_exp5.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_exp6.pickle", "r") as f:
        result1 = pickle.load(f)

    ##wrap and normalize ##
    medians0, iqrs0 = wrap_repeat(result0)
    medians1, iqrs1 = wrap_repeat(result1)
    # medians0 = rescale(medians0)
    # iqrs0 = rescale(iqrs0)
    # medians1 = rescale(medians1)
    # iqrs1 = rescale(iqrs1)
    #################



    line, = plt.plot(medians0['x'], medians0["linear_review"], label="linear_review", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["linear_review"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="P_U_S_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="H_C_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["continuous_aggressive"], label="P_C_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"], label="H_U_S_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_aggressive"], label="H_C_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="H_U_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["continuous_active"], label="P_C_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs1["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="P_U_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())
    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.95, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/comp_repeat_exp" + str(id) + ".eps")
    plt.savefig("../figure/comp_repeat_exp" + str(id) + ".png")


def draw_margin(id):
    margins = ["0.8", "0.9", "1.0", "1.2", "1.5", "2.0"]

    N=len(margins)


    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}
    plt.rcParams.update(paras)

    scalarMap = colorcode(N)
    result={}
    medians={}
    iqrs={}
    for margin in margins:
        with open("../dump/repeat_margin_"+margin+".pickle", "r") as f:
            result[margin]=pickle.load(f)
        ##wrap and normalize ##
        medians[margin], iqrs[margin] = wrap_repeat(result[margin])
        # medians[margin] = rescale(medians[margin])
        # iqrs[margin] = rescale(iqrs[margin])
        #################

    for ind,margin in enumerate(margins):
        line, = plt.plot(medians[margin]['x'], medians[margin]["semi_continuous_aggressive"], label=margin, color = scalarMap.to_rgba(ind))
        plt.plot(iqrs[margin]['x'], iqrs[margin]["linear_review"], "-.", color=line.get_color())

    plt.ylabel("Relevant Found")
    plt.xlabel("Documents Reviewed")
    plt.legend(bbox_to_anchor=(0.95, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/margin_exp" + str(id) + ".eps")
    plt.savefig("../figure/margin_exp" + str(id) + ".png")





def IST_comp_draw(set):

    N= 13

    indices = range(N)

    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    scalarMap = colorcode(N)

    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_"+set+"_5.pickle", "r") as f:
        result1 = pickle.load(f)


    ##wrap and normalize ##


    medians0, iqrs0 = wrap_repeat(result0)
    medians1, iqrs1 = wrap_repeat(result1)

    posnum = medians0['simple_active'][-1]
    docnum = medians0['x'][-1]

    medians0 = rescaleY(medians0,posnum)
    iqrs0 = rescaleY(iqrs0,posnum)
    medians1 = rescaleY(medians1,posnum)
    iqrs1 = rescaleY(iqrs1,posnum)
    #################

    ###### cut ######
    Display = 250
    medians0 = cutListinDict(medians0,Display)
    medians1 = cutListinDict(medians1,Display)
    iqrs0 = cutListinDict(iqrs0,Display)
    iqrs1 = cutListinDict(iqrs1,Display)

    #################




    line, = plt.plot(medians1['x'], medians1["simple_active"], label="P_U_S_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["simple_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="P_U_S_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["continuous_active"], label="P_C_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="P_C_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous"], label="P_U_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["semi_continuous"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="P_U_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())

    line, = plt.plot(medians1['x'], medians1["linear_review"], label="linear_review", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs1['x'], iqrs1["linear_review"], "-.", color=line.get_color())


    line, = plt.plot(medians0['x'], medians0["simple_active"], label="H_U_S_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["simple_active"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"], label="H_U_S_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="H_C_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="H_C_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous"], label="H_U_C_N", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["semi_continuous"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="H_U_C_A", color = scalarMap.to_rgba(indices.pop()))
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.90), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/IST_comp_" + set + ".eps")
    plt.savefig("../figure/IST_comp_" + set + ".png")


def IST_dom_draw(set):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_"+set+"_5.pickle", "r") as f:
        result1 = pickle.load(f)

    ##wrap and normalize ##


    medians0, iqrs0 = wrap_repeat(result0)
    medians1, iqrs1 = wrap_repeat(result1)

    posnum = medians0['simple_active'][-1]
    docnum = medians0['x'][-1]

    medians0 = rescaleY(medians0,posnum)
    iqrs0 = rescaleY(iqrs0,posnum)
    medians1 = rescaleY(medians1,posnum)
    iqrs1 = rescaleY(iqrs1,posnum)
    #################

    ###### cut ######
    Display = 250
    medians0 = cutListinDict(medians0,Display)
    medians1 = cutListinDict(medians1,Display)
    iqrs0 = cutListinDict(iqrs0,Display)
    iqrs1 = cutListinDict(iqrs1,Display)

    #################


    ### start with P_U_S_A (Patient Active Learning), compare last code first ##
    ### P_U_S_A vs. P_U_S_N ####
    plt.figure(4)


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="P_U_S_A")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["simple_active"], label="P_U_S_N")
    plt.plot(iqrs1['x'], iqrs1["simple_active"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_4_" + set + ".eps")
    plt.savefig("../figure/IST_4_" + set + ".png")

    ### compare third code ##
    ### P_U_S_A vs. P_U_C_A ####
    plt.figure(3)


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="P_U_S_A")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="P_U_C_A")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_3_" + set + ".eps")
    plt.savefig("../figure/IST_3_" + set + ".png")

    ### compare second code ##
    ### P_U_C_A vs. P_C_C_A ####
    plt.figure(2)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="P_U_C_A")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="P_C_C_A")
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_2_" + set + ".eps")
    plt.savefig("../figure/IST_2_" + set + ".png")

    ### compare first code ##
    ### P_U_C_A vs. P_C_C_A vs. H_U_C_A vs. H_C_C_A####
    plt.figure(1)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="P_U_C_A")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="P_C_C_A")
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())


    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="H_U_C_A")
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="H_C_C_A")
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_1_" + set + ".eps")
    plt.savefig("../figure/IST_1_" + set + ".png")

    ### compare best with baselines ##
    ### H_C_C_A vs. H_C_C_N vs. P_U_S_A ####
    plt.figure(0)


    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="H_C_C_A")
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="H_C_C_N")
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())

    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="P_U_S_A")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_0_" + set + ".eps")
    plt.savefig("../figure/IST_0_" + set + ".png")






def numbers(set):
    
    target = 0.9
    
    
    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        result0=pickle.load(f)
    with open("../dump/repeat_"+set+"_5.pickle", "r") as f:
        result1=pickle.load(f)

    medians0, iqrs0 = wrap_repeat(result0)

    posnum = medians0['simple_active'][-1]
    
    thres = int(target*posnum)

    methods=['new_continuous_aggressive','continuous_active','aggressive_undersampling','semi_continuous_aggressive','simple_active','semi_continuous']
    tests=[]
    for method in methods:
        wheres1=['H_'+method]
        for value in result0:
            tmp=thres
            best=value[method]
            while True:
                try:
                    where = value['x'][best.index(tmp)]
                    break
                except:
                    tmp=tmp+1
            wheres1.append(where)
        wheres2=['P_'+method]
        for value in result1:
            tmp=thres
            best=value[method]
            while True:
                try:
                    where = value['x'][best.index(tmp)]
                    break
                except:
                    tmp=tmp+1
            wheres2.append(where)
        tests.append(wheres1)
        tests.append(wheres2)

    rdivDemo(tests)

    set_trace()







##### UPDATE exp
def update_exp():
    stepsize=10
    with open("../dump/Hall2007.pickle","rb") as handle:
        csr_mat1 = pickle.load(handle)
        labels1 = pickle.load(handle)
        vocab1 = pickle.load(handle)
    with open("../dump/Hall2010.pickle","rb") as handle:
        csr_mat2 = pickle.load(handle)
        labels2 = pickle.load(handle)
        vocab2 = pickle.load(handle)
    with open("../dump/ieee.pickle","rb") as handle:
        csr_mat3 = pickle.load(handle)
        labels3 = pickle.load(handle)
        vocab3 = pickle.load(handle)

    update_exps(csr_mat1,labels1,csr_mat2,labels2,csr_mat3,labels3,vocab2,vocab3,stepsize=stepsize)

def update_exps(csr_mat1,labels1,csr_mat2,labels2,csr_mat3,labels3,vocab2,vocab3,stepsize=10):
    result, train = simple_hcca1(csr_mat1, labels1, step=stepsize ,initial=10, pos_limit=1, thres=20)
    result2, model2 = simple_hcca2(csr_mat2, labels2, csr_mat1[train], labels1[train], step=stepsize)
    model2=model_transform(model2,vocab2,vocab3)
    result3, model3 = simple_hcca3(csr_mat3, labels3, model2, step=stepsize ,initial=10, pos_limit=1, thres=30)
    return {"Hall2007": result, "Hall2010": result2, "ieee": result3}


def model_transform(model,vocab,vocab_new):
    w=[]
    for term in vocab_new:
        try:
            ind=vocab.index(term)
            w.append(model['w'][0,ind])
        except:
            w.append(0)
    model['w']=np.array(w)
    return model

def simple_hcca1(csr_mat, labels, step=10 ,initial=10, pos_limit=1, thres=30, stop=0.9):
    num=len(labels)
    pool=range(num)
    train=[]
    steps = np.array(range(int(num / step))) * step

    pos=0
    pos_track=[0]
    clf = svm.SVC(kernel='linear', probability=True)
    begin=0
    result={}
    enough=False

    total=Counter(labels)["yes"]*stop


    for idx, round in enumerate(steps[:-1]):

        if pos >= total:
            if enough:
                pos_track_f=pos_track9
                train_f=train9
            elif begin:
                pos_track_f=pos_track4
                train_f=train4
            else:
                pos_track_f=pos_track
                train_f=train
            break

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
            pool4 = pool2[:]
            train4 = train2[:]
            pos_track4 = pos_track2[:]
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
                if pos>=thres:
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

        print("Round #{id} passed\r".format(id=round), end="")

    result["begin"] = begin
    result["x"] = steps[:len(pos_track_f)]
    result["new_continuous_aggressive"] = pos_track_f
    return result, train_f

def simple_hcca2(csr_mat, labels, csr_old, labels_old, step=10, stop=0.9):
    num=len(labels)
    pool=range(num)
    total=Counter(labels)["yes"]*stop
    already=Counter(labels_old)["yes"]
    train=range(num,num+len(labels_old))
    labels=np.array(labels.tolist()+labels_old.tolist())
    csr_mat=csr_matrix(csr_mat.todense().tolist()+csr_old.todense().tolist())
    steps = np.array(range(int(num / step))) * step

    pos=0
    pos_track9=[0]
    clf = svm.SVC(kernel='linear', probability=True)
    result={}




    for idx, round in enumerate(steps[:-1]):

        if pos >= total:
            break

        clf.fit(csr_mat[train], labels[train])
        poses = np.where(labels[train] == "yes")[0]
        negs = np.where(labels[train] == "no")[0]
        train_dist = clf.decision_function(csr_mat[train][negs])
        negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
        sample9 = np.array(train)[poses].tolist() + np.array(train)[negs][negs_sel].tolist()

        clf.fit(csr_mat[sample9], labels[sample9])
        pred_proba9 = clf.predict_proba(csr_mat[pool])
        pos_at = list(clf.classes_).index("yes")
        proba9 = pred_proba9[:, pos_at]
        sort_order_certain9 = np.argsort(1 - proba9)
        can9 = [pool[i] for i in sort_order_certain9[:step]]
        train.extend(can9)
        pool9 = list(set(pool) - set(can9))
        pos = Counter(labels[train])["yes"]-already
        pos_track9.append(pos)

        print("Round #{id} passed\r".format(id=round), end="")


    result["x"] = steps[:len(pos_track9)]
    result["new_continuous_aggressive"] = pos_track9
    clf.fit(csr_mat[train], labels[train])
    if labels[train][0]=="yes":
        pos_at=0
    else:
        pos_at=1
    w=clf.coef_
    model={'w': w,'pos_at': pos_at}
    return result, model


def simple_hcca3(csr_mat, labels, model, step=10 ,initial=10, pos_limit=1, thres=30, stop=0.9):
    num=len(labels)
    pool=range(num)
    train=[]
    steps = np.array(range(int(num / step))) * step

    pos=0
    pos_track=[0]
    clf = svm.SVC(kernel='linear', probability=True)
    begin=0
    result={}
    enough=False

    total=Counter(labels)["yes"]*stop


    for idx, round in enumerate(steps[:-1]):

        if pos >= total:
            if enough:
                pos_track_f=pos_track9
                train_f=train9
            elif begin:
                pos_track_f=pos_track4
                train_f=train4
            else:
                pos_track_f=pos_track
                train_f=train
            break

        order = np.argsort(model['w']*csr_mat[pool].transpose())
        if model['pos_at'] == 1:
            can=[pool[i] for i in order[-step:]]
        else:
            can=[pool[i] for i in order[:step]]
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
            pool4 = pool2[:]
            train4 = train2[:]
            pos_track4 = pos_track2[:]
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
                if pos>=thres:
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

        print("Round #{id} passed\r".format(id=round), end="")

    result["begin"] = begin
    result["x"] = steps[:len(pos_track_f)]
    result["new_continuous_aggressive"] = pos_track_f
    clf.fit(csr_mat[train_f], labels[train_f])
    if labels[train_f][0]=="yes":
        pos_at=0
    else:
        pos_at=1
    w=clf.coef_
    model={'w': w,'pos_at': pos_at}
    return result, model



if __name__ == "__main__":
    eval(cmd())
