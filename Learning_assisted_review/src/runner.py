from __future__ import division, print_function



# from ES_CORE import ESHandler
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
from sklearn.cluster import KMeans
import lda
from sklearn.decomposition import LatentDirichletAllocation
from time import time


#
# es = ESHandler(force_injest=False)
# container = Vessel(
#         OPT=None,
#         SVM=None,
#         round=0
# )

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


def export_txt(set):
    es=ESHandler(es=defaults(TYPE_NAME=set),force_injest=False)
    res=es.get_unlabeled()
    den="\t"
    txt=[]
    txt_content='Document Title'+den+'Abstract'+den+'Year'+den+'PDF Link'+den+'label\n'
    txt.append(txt_content)
    for x in res['hits']['hits']:
        txt_content=x['_source']['title']+den+x['_source']['abstract']+den+x['_source']['year']+den+x['_source']['ft_url']+den+x['_source']['user']+"\n"
        txt.append(unicodedata.normalize('NFKD', txt_content).encode('ascii', 'ignore'))
    with open("../dump/" + str(set) + ".txt","w") as f:
        f.writelines(txt)

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


def IST_split_draw(set):


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


    plt.figure(0)

    line, = plt.plot(medians1['x'], medians1["simple_active"], label="$PUS\\bar{N}$")
    plt.plot(iqrs1['x'], iqrs1["simple_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["continuous_active"], label="$P\\bar{U}\\bar{S}\\bar{A}$")
    plt.plot(iqrs1['x'], iqrs1["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$")
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous"], label="$PU\\bar{S}\\bar{A}$")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())

    line, = plt.plot(medians1['x'], medians1["linear_review"], label="linear_review")
    plt.plot(iqrs1['x'], iqrs1["linear_review"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')

    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/IST_P_" + set + ".eps")
    plt.savefig("../figure/IST_P_" + set + ".png")



    plt.figure(1)
    line, = plt.plot(medians0['x'], medians0["simple_active"], label="$\\bar{P}US\\bar{A}$")
    plt.plot(iqrs0['x'], iqrs0["simple_active"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"], label="$\\bar{P}USA$")
    plt.plot(iqrs0['x'], iqrs0["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$")
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$")
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous"], label="$\\bar{P}U\\bar{S}\\bar{A}$")
    plt.plot(iqrs0['x'], iqrs0["semi_continuous"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="$\\bar{P}U\\bar{S}A$")
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())

    line, = plt.plot(medians0['x'], medians0["linear_review"], label="linear_review")
    plt.plot(iqrs0['x'], iqrs0["linear_review"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/IST_H_" + set + ".eps")
    plt.savefig("../figure/IST_H_" + set + ".png")

def IST_split_draw_noiqr(set):


    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 30, 'axes.labelsize': 30, 'legend.frameon': False,
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


    plt.figure(0)

    line, = plt.plot(medians1['x'], medians1["simple_active"], label="$PUS\\bar{N}$")
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$")
    line, = plt.plot(medians1['x'], medians1["continuous_active"], label="$P\\bar{U}\\bar{S}\\bar{A}$")

    line, = plt.plot(medians1['x'], medians1["linear_review"], label="linear_review")

    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$")
    line, = plt.plot(medians1['x'], medians1["semi_continuous"], label="$PU\\bar{S}\\bar{A}$")
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$")



    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')

    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/IST_P_" + set + ".eps")
    plt.savefig("../figure/IST_P_" + set + ".png")



    plt.figure(1)
    line, = plt.plot(medians0['x'], medians0["simple_active"], label="$\\bar{P}US\\bar{A}$")
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"], label="$\\bar{P}USA$")
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$")

    line, = plt.plot(medians0['x'], medians0["linear_review"], label="linear_review")

    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$")
    line, = plt.plot(medians0['x'], medians0["semi_continuous"], label="$\\bar{P}U\\bar{S}\\bar{A}$")
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="$\\bar{P}U\\bar{S}A$")




    # plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/IST_H_" + set + ".eps")
    plt.savefig("../figure/IST_H_" + set + ".png")


def IST_fade_draw(set):

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




    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$ (FASTREAD)")
    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$ (EBM)")
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$ (TAR)")
    line, = plt.plot(medians1['x'], medians1["simple_active"],'.', color='gray', label="$PUS\\bar{N}$")
    line, = plt.plot(medians1['x'], medians1["continuous_active"],'.', color='gray', label="$P\\bar{U}\\bar{S}\\bar{A}$")
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"],'.', color='gray', label="$P\\bar{U}\\bar{S}A$")

    line, = plt.plot(medians1['x'], medians1["linear_review"],'.', color='gray', label="linear_review")


    line, = plt.plot(medians1['x'], medians1["semi_continuous"],'.', color='gray', label="$PU\\bar{S}\\bar{A}$")
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"],'.', color='gray', label="$PU\\bar{S}A$")
    line, = plt.plot(medians0['x'], medians0["simple_active"],'.', color='gray', label="$\\bar{P}US\\bar{A}$")
    line, = plt.plot(medians0['x'], medians0["aggressive_undersampling"],'.', color='gray', label="$\\bar{P}USA$")
    line, = plt.plot(medians0['x'], medians0["semi_continuous"],'.', color='gray', label="$\\bar{P}U\\bar{S}\\bar{A}$")
    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"],'.', color='gray', label="$\\bar{P}U\\bar{S}A$")


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="red",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="red",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
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


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["simple_active"], label="$PUS\\bar{A}$", color="red")
    plt.plot(iqrs1['x'], iqrs1["simple_active"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_4_" + set + ".eps")
    plt.savefig("../figure/IST_4_" + set + ".png")

    ### compare third code ##
    ### P_U_S_A vs. P_U_C_A ####
    plt.figure(3)


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$", color="red")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_3_" + set + ".eps")
    plt.savefig("../figure/IST_3_" + set + ".png")

    ### compare second code ##
    ### P_U_C_A vs. P_C_C_A ####
    plt.figure(2)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$", color="red")
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())

    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_2_" + set + ".eps")
    plt.savefig("../figure/IST_2_" + set + ".png")

    ### compare first code ##
    ### P_U_C_A vs. P_C_C_A vs. H_U_C_A vs. H_C_C_A####
    plt.figure(1)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$")
    plt.plot(iqrs1['x'], iqrs1["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$")
    plt.plot(iqrs1['x'], iqrs1["new_continuous_aggressive"], "-.", color=line.get_color())


    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="$\\bar{P}U\\bar{S}A$")
    plt.plot(iqrs0['x'], iqrs0["semi_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$")
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="black",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_1_" + set + ".eps")
    plt.savefig("../figure/IST_1_" + set + ".png")

    ### compare best with baselines ##
    ### H_C_C_A vs. H_C_C_N vs. P_U_S_A ####
    plt.figure(0)


    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$ (FASTREAD)")
    plt.plot(iqrs0['x'], iqrs0["new_continuous_aggressive"], "-.", color=line.get_color())
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$ (Continuous Active Learning)")
    plt.plot(iqrs0['x'], iqrs0["continuous_active"], "-.", color=line.get_color())

    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$ (Patient Active Learning)")
    plt.plot(iqrs1['x'], iqrs1["aggressive_undersampling"], "-.", color=line.get_color())


    plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="black",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_0_" + set + ".eps")
    plt.savefig("../figure/IST_0_" + set + ".png")

def IST_dom_draw_noiqr(set):
    font = {'family': 'default',
            # 'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
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

    lines=['-','--','-.',':']
    ### start with P_U_S_A (Patient Active Learning), compare last code first ##
    ### P_U_S_A vs. P_U_S_N ####
    plt.figure(4)


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$",linestyle=lines[0])
    line, = plt.plot(medians1['x'], medians1["simple_active"], label="$PUS\\bar{A}$", color="red",linestyle=lines[1])

    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_4_" + set + ".eps")
    plt.savefig("../figure/IST_4_" + set + ".png")

    ### compare third code ##
    ### P_U_S_A vs. P_U_C_A ####
    plt.figure(3)


    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$",linestyle=lines[0])
    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$", color="red",linestyle=lines[1])

    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_3_" + set + ".eps")
    plt.savefig("../figure/IST_3_" + set + ".png")

    ### compare second code ##
    ### P_U_C_A vs. P_C_C_A ####
    plt.figure(2)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$",linestyle=lines[0])
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$", color="red",linestyle=lines[1])

    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_2_" + set + ".eps")
    plt.savefig("../figure/IST_2_" + set + ".png")

    ### compare first code ##
    ### P_U_C_A vs. P_C_C_A vs. H_U_C_A vs. H_C_C_A####
    plt.figure(1)

    line, = plt.plot(medians1['x'], medians1["semi_continuous_aggressive"], label="$PU\\bar{S}A$",linestyle=lines[0])
    line, = plt.plot(medians1['x'], medians1["new_continuous_aggressive"], label="$P\\bar{U}\\bar{S}A$",linestyle=lines[1],color="red")


    line, = plt.plot(medians0['x'], medians0["semi_continuous_aggressive"], label="$\\bar{P}U\\bar{S}A$",linestyle=lines[2],color="green")
    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$",linestyle=lines[3],color="brown")


    # plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="black",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_1_" + set + ".eps")
    plt.savefig("../figure/IST_1_" + set + ".png")

    ### compare best with baselines ##
    ### H_C_C_A vs. H_C_C_N vs. P_U_S_A ####
    plt.figure(0)


    line, = plt.plot(medians0['x'], medians0["new_continuous_aggressive"], label="$\\bar{P}\\bar{U}\\bar{S}A$ (FASTREAD)",linestyle=lines[0])
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$ (Continuous Active Learning)",linestyle=lines[1],color="red")

    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$ (Patient Active Learning)",linestyle=lines[2],color="green")


    # plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="black",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_0_" + set + ".eps")
    plt.savefig("../figure/IST_0_" + set + ".png")

    ### compare baselines with linear review ##
    ### H_C_C_N vs. P_U_S_A vs. linear review ####
    plt.figure(10)


    line, = plt.plot(medians0['x'], medians0["linear_review"], label="Linear Review",linestyle=lines[0])
    line, = plt.plot(medians0['x'], medians0["continuous_active"], label="$\\bar{P}\\bar{U}\\bar{S}\\bar{A}$ (Continuous Active Learning)",linestyle=lines[1],color="red")

    line, = plt.plot(medians1['x'], medians1["aggressive_undersampling"], label="$PUSA$ (Patient Active Learning)",linestyle=lines[2],color="green")



    # plt.plot(medians0['x'][medians0['stable']], medians0["simple_active"][medians0['stable']], color="black",marker='o')
    plt.plot(medians0['x'][medians0['begin']], medians0["simple_active"][medians0['begin']], color="white", marker='o')
    # plt.plot(medians1['x'][medians1['stable']], medians1["simple_active"][medians1['stable']], color="black",marker='o')
    plt.plot(medians1['x'][medians1['begin']], medians1["simple_active"][medians1['begin']], color="white", marker='o')


    tick = 500
    x=[i*500 for i in xrange(int(docnum/tick)) if i*500<= int(Display*10)]


    xlabels = [str(z)+"\n("+'%.1f'%(z/docnum*100)+"%)" for z in x]

    plt.xticks(x, xlabels)

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/IST_B_" + set + ".eps")
    plt.savefig("../figure/IST_B_" + set + ".png")


def PoR(set):
    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        result=pickle.load(f)
    x=[]
    pos=[]
    yy=[]
    for rep in result:
        y=[rep["new_continuous_aggressive"][ind]/rep["x"][ind] if rep["x"][ind]>0 else 0 for ind in xrange(len(rep["x"]))]
        yy.append(y)
        tmp=np.argmax(y)
        x.append(rep["x"][tmp])
        pos.append(rep["new_continuous_aggressive"][tmp])
    sup=np.median(yy,axis=0)
    iqr=np.percentile(yy,75,axis=0)-np.percentile(yy,25,axis=0)

    font = {'family': 'default',
            # 'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    plt.figure(0)

    line, =plt.plot(result[0]['x'][:90], sup[:90])
    plt.plot(result[0]['x'][:90], iqr[:90],"-.",color=line.get_color())

    plt.ylabel(set+"\nRetrieval Rate per Cost")
    plt.xlabel("Studies Reviewed")
    # plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/PoR_"+set+".eps")
    plt.savefig("../figure/PoR_"+set+".png")

def PoR2():
    sets=["Wahono","Hall"]
    data={}
    for set in sets:
        data[set]={}
        with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
            results=pickle.load(f)
        pos=results[0]['new_continuous_aggressive'][-1]
        x=range(pos+1)
        y=[]
        z=[]
        q=[]
        for target in x:
            yyy=[]
            zzz=[]
            for result in results:
                for i in xrange(len(result['x'])):
                    if result['new_continuous_aggressive'][i]>= target:
                        tmp=i
                        break
                yy=result['x'][tmp]
                zz=target/yy if yy>0 else 0
                yyy.append(yy)
                zzz.append(zz)
            z.append(np.median(zzz))
            q.append(np.percentile(zzz,75)-np.percentile(zzz,25))
            y.append(np.median(yyy))
        data[set]['x']=np.array(x)/pos
        data[set]['y']=y
        data[set]['z']=z
        data[set]['q']=q


    font = {'family': 'default',
            # 'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    plt.figure(0)
    for set in sets:
        line, =plt.plot(data[set]['x'], data[set]['z'], label=set)
        # plt.plot(data[set]['x'], data[set]['q'],"-.",color=line.get_color())


    x=np.array(range(11))*0.1
    plt.xticks(x,x)

    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/PoR.eps")
    plt.savefig("../figure/PoR.png")

    plt.figure(1)
    for set in sets:
        line, =plt.plot(data[set]['x'], data[set]['y'], label=set)
        # plt.plot(data[set]['x'], data[set]['q'],"-.",color=line.get_color())


    x=np.array(range(11))*0.1
    plt.xticks(x,x)

    plt.ylabel("Studies Reviewed")
    plt.xlabel("Recall")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/PoR2.eps")
    plt.savefig("../figure/PoR2.png")






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
    decode={'new_continuous_aggressive':'\\bar{U}\\bar{S}A','continuous_active':'\\bar{U}\\bar{S}\\bar{A}','aggressive_undersampling':'USA','semi_continuous_aggressive':'U\\bar{S}A','simple_active':'US\\bar{A}','semi_continuous':'U\\bar{S}\\bar{A}'}
    tests=[]
    for method in methods:
        wheres1=['$\\bar{P}'+decode[method]+'$']
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
        wheres2=['$P'+decode[method]+'$']
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

    rdivDemo(tests, isLatex=True)

    set_trace()

##### Draw percentile
def draw_percentile(set):
    font = {'family': 'default',
            # 'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        results=pickle.load(f)
    pos_num=results[0]['simple_active'][-1]
    stats=percentile(results)

    colors=['red','blue','green','cyan', 'purple']
    plt.figure(0)
    # for i,ind in enumerate(stats['new_continuous_aggressive']):
    #     plt.plot(results[0]['x'][:len(stats["new_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["new_continuous_aggressive"][ind]),color=colors[i],label=str(ind)+"th Percentile of $\\bar{P}\\bar{U}\\bar{S}A$")
    # for i,ind in enumerate(stats['semi_continuous_aggressive']):
    #     plt.plot(results[0]['x'][:len(stats["semi_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["semi_continuous_aggressive"][ind]), "-.", color=colors[i], label=str(ind)+"th Percentile of $\\bar{P}U\\bar{S}A$")

    for i,ind in enumerate(np.sort(stats['new_continuous_aggressive'].keys())):
        plt.plot(results[0]['x'][:len(stats["new_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["new_continuous_aggressive"][ind]),color=colors[i],label=str(ind)+"th Percentile")
        plt.plot(results[0]['x'][:len(stats["semi_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["semi_continuous_aggressive"][ind]), "-.", color=colors[i])

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    # plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0.7, 0.80), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_"+set+".eps")
    plt.savefig("../figure/percentile_"+set+".png")

def draw_HCCA(set):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        results=pickle.load(f)
    pos_num=results[0]['simple_active'][-1]
    stats=percentile(results)

    colors=['red','blue','green','cyan', 'purple']
    plt.figure(0)
    for i,ind in enumerate(np.sort(stats['new_continuous_aggressive'].keys())):
        plt.plot(results[0]['x'][:len(stats["new_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["new_continuous_aggressive"][ind]),color=colors[i],label=str(ind)+"th Percentile")

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.85), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_best_"+set+".eps")
    plt.savefig("../figure/percentile_best_"+set+".png")

def draw_HUCA(set):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 25, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/repeat_"+set+"_1.pickle", "r") as f:
        results=pickle.load(f)
    pos_num=results[0]['simple_active'][-1]
    stats=percentile(results)

    colors=['red','blue','green','cyan', 'purple']
    plt.figure(0)
    for i,ind in enumerate(np.sort(stats['semi_continuous_aggressive'].keys())):
        plt.plot(results[0]['x'][:len(stats["semi_continuous_aggressive"][ind])], map(lambda x: x/pos_num, stats["semi_continuous_aggressive"][ind]),color=colors[i],label=str(ind)+"th Percentile")

    plt.ylabel(set+"\nRecall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.85), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/percentile_second_"+set+".eps")
    plt.savefig("../figure/percentile_second_"+set+".png")


def percentile(results):
    stats={}
    methods=['new_continuous_aggressive','continuous_active','aggressive_undersampling','semi_continuous_aggressive','simple_active','semi_continuous']
    tests={}
    thres=int(0.9*results[0]["simple_active"][-1])
    for method in methods:
        tests[method]=[]
        for value in results:
            tmp=thres
            best=value[method]
            while True:
                try:
                    where = best.index(tmp)
                    break
                except:
                    tmp=tmp+1
            tests[method].append(where)
    for k in methods:
        stats[k]={}
        tmp = np.array([what[k] for what in results])

        order=np.argsort(tests[k])
        for ind in [0,50,75,90,100]:
            nth=order[int(ind*(len(order)-1)/100)]
            stats[k][ind]=tmp[nth][:tests[k][nth]]
    return stats


##### Draw UPDATE


def update_repeat_draw(id):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/update_repeat"+str(id)+".pickle", "r") as f:
        results=pickle.load(f)

    # medians,iqrs=wrap_repeat_update(results)
    stats=bestNworst(results)

    # for i,key in enumerate(medians):
    #     plt.figure(i)
    #     line, = plt.plot(medians[key]['x'], medians[key]["new_continuous_aggressive"])
    #     plt.plot(iqrs[key]['x'], iqrs[key]["new_continuous_aggressive"], "-.", color=line.get_color())
    #
    #     # plt.plot(results[key]['x'][results[key]['begin']], results[key]["new_continuous_aggressive"][results[key]['begin']], color="white", marker='o')
    #
    #     plt.ylabel("Retrieval Rate")
    #     plt.xlabel("Studies Reviewed")
    #     plt.savefig("../figure/update_repeat_"+key+".eps")
    #     plt.savefig("../figure/update_repeat_"+key+".png")

    for i,key in enumerate(stats):
        plt.figure(10+i)
        for ind in stats[key]['x']:
            plt.plot(stats[key]['x'][ind], stats[key]["new_continuous_aggressive"][ind],label=str(ind)+"th Percentile")


        plt.ylabel("Retrieval Rate")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/update_bestNworst_"+key+str(id)+".eps")
        plt.savefig("../figure/update_bestNworst_"+key+str(id)+".png")

def bestNworst(results):
    stats={}

    for key in results[0]:
        stats[key]={}
        for k in results[0][key].keys():
            stats[key][k]={}
            tmp = np.array([what[key][k] for what in results])
            if k=="begin":
                continue
            order=np.argsort([len(seq) for seq in tmp])
            for ind in [0,25,50,75,100]:
                stats[key][k][ind]=tmp[order[int(ind*(len(order)-1)/100)]]


    return stats

# def wrap_repeat_update(results):
#     medians={}
#     iqrs={}
#     for key in results[0].keys():
#         medians[key]={}
#         iqrs[key]={}
#         for k in results[0][key].keys():
#             tmp = np.array([what[key][k] for what in results])
#             if k == 'x':
#                 shortest=np.min([len(seq) for seq in tmp])
#                 medians[key][k]=iqrs[key][k]=tmp[0][:shortest]
#
#             elif k == 'begin':
#                 continue
#             else:
#                 tmp2=[t[:shortest] for t in tmp]
#                 medians[key][k] = np.median(tmp2,axis=0)
#                 iqrs[key][k] = np.percentile(tmp2,75,axis=0) - np.percentile(tmp2,25,axis=0)
#     return medians, iqrs


## LDA ##

"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat

def comp_LDA(tp):
    with open("../dump/ieee.pickle","rb") as handle:
        csr_mat3 = pickle.load(handle)
        labels3 = pickle.load(handle)
        vocab3 = pickle.load(handle)
    if container.SVM is None:
        container.also(SVM=SVM(disp=stepsize, set="ieee", opt=container.OPT).featurize())
    csr,labels = container.SVM.extract_data()

    lda1 = lda.LDA(n_topics=int(tp), alpha=0.1, eta=0.01, n_iter=200)

    # lda2 = LatentDirichletAllocation(n_topics=int(tp), learning_method='online', doc_topic_prior=0.1, topic_word_prior=0.01, max_iter=200)
    time1=time()
    csr_mat4 = csr_matrix(lda1.fit_transform(csr))
    csr_mat4 = l2normalize(csr_mat4)
    n_top_words = 8
    for i, topic_dist in enumerate(lda1.topic_word_):
        topic_words = np.array(vocab3)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # csr_mat4 = csr_matrix(lda2.fit_transform(csr_mat3))
    time2=time()-time1
    print(time2)
    # tops = lda1.doc_topic_
    # topic_word = lda1.topic_word_
    # topic_word=lda2.components_
    result, train = hcca_lda(csr_mat3, csr_mat4, labels3, step=stepsize ,initial=10, pos_limit=1, thres=20)
    with open("../dump/lda"+str(tp)+".pickle","wb") as handle:
        pickle.dump(result,handle)
    set_trace()

def hcca_lda(csr_mat,csr_lda, labels, step=10 ,initial=10, pos_limit=1, thres=30, stop=0.9):
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

    # total = 1000

    for idx, round in enumerate(steps[:-1]):

        if round >= 2500:
            if enough:
                pos_track_f=pos_track9
                train_f=train9
                pos_track_l=pos_track8
                train_l=train8
            elif begin:
                pos_track_f=pos_track4
                train_f=train4
                pos_track_l=pos_track2
                train_l=train2
            else:
                pos_track_f=pos_track
                train_f=train
                pos_track_l=pos_track
                train_l=train
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

            ## lda
            clf.fit(csr_lda[train2], labels[train2])
            pred_proba2 = clf.predict_proba(csr_lda[pool2])
            pos_at = list(clf.classes_).index("yes")
            proba2 = pred_proba2[:, pos_at]
            sort_order_certain2 = np.argsort(1 - proba2)
            can2 = [pool2[i] for i in sort_order_certain2[:step]]
            train2.extend(can2)
            pool2 = list(set(pool2) - set(can2))
            pos = Counter(labels[train2])["yes"]
            pos_track2.append(pos)


            ################ new *_C_C_A
            if not enough:
                if pos>=thres:
                    enough=True
                    pos_track9=pos_track4[:]
                    train9=train4[:]
                    pool9=pool4[:]
                    pos_track8=pos_track2[:]
                    train8=train2[:]
                    pool8=pool2[:]
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

                clf.fit(csr_lda[train8], labels[train8])
                poses = np.where(labels[train8] == "yes")[0]
                negs = np.where(labels[train8] == "no")[0]
                train_dist = clf.decision_function(csr_lda[train8][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample8 = np.array(train8)[poses].tolist() + np.array(train8)[negs][negs_sel].tolist()

                clf.fit(csr_lda[sample8], labels[sample8])
                pred_proba8 = clf.predict_proba(csr_lda[pool8])
                pos_at = list(clf.classes_).index("yes")
                proba8 = pred_proba8[:, pos_at]
                sort_order_certain8 = np.argsort(1 - proba8)
                can8 = [pool8[i] for i in sort_order_certain8[:step]]
                train8.extend(can8)
                pool8 = list(set(pool8) - set(can8))
                pos = Counter(labels[train8])["yes"]
                pos_track8.append(pos)

        print("Round #{id} passed\r".format(id=round), end="")

    result["begin"] = begin
    result["x"] = steps[:len(pos_track_f)]
    result["new_continuous_aggressive"] = pos_track_f
    result["lda"] = pos_track_l
    return result, train_f

def draw_LDA(id):
    with open("../dump/lda"+str(id)+".pickle","rb") as handle:
        result = pickle.load(handle)
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)
    plt.figure(0)
    plt.plot(result['x'], result['lda'],label="lda")
    plt.plot(result['x'], result["new_continuous_aggressive"],label="no_lda")

    plt.ylabel("Recall")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/lda_"+str(id)+".eps")
    plt.savefig("../figure/lda_"+str(id)+".png")



##### UPDATE exp
def update_exp(id):
    repeats=30
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
    results=[]
    for i in xrange(repeats):
        result=update_exps(csr_mat1,labels1,csr_mat2,labels2,csr_mat3,labels3,vocab2,vocab3,stepsize=stepsize)
        results.append(result)

    with open("../dump/update_repeat"+str(id)+".pickle","wb") as handle:
        pickle.dump(results,handle)

def update_exps(csr_mat1,labels1,csr_mat2,labels2,csr_mat3,labels3,vocab2,vocab3,stepsize=10):
    t1=time()
    result, train = simple_hcca1(csr_mat1, labels1, step=stepsize ,initial=10, pos_limit=1, thres=20)
    result2, model2 = simple_hcca2(csr_mat2, labels2, csr_mat1[train], labels1[train], step=stepsize)
    t2=time()-t1
    print(t2)
    model2=model_transform(model2,vocab2,vocab3)
    t1=time()
    result3, model3 = simple_hcca3(csr_mat3, labels3, model2, step=stepsize ,initial=50, pos_limit=5, thres=30)
    t2=time()-t1
    print(t2)
    result5, model5 = simple_hcca3(csr_mat3, labels3, model2, step=stepsize ,initial=10, pos_limit=2, thres=30,clustering=True,sample=2)
    result4, train4=simple_hcca1(csr_mat2, labels2, step=stepsize ,initial=10, pos_limit=1, thres=20)
    return {"Hall2007": result, "Hall2010": result2, "ieee": result3, "Hall2010init": result4, "ieee_clustering": result5}


def model_transform(model,vocab,vocab_new):
    w=[]
    for term in vocab_new:
        try:
            ind=vocab.index(term)
            w.append(model['w'][0,ind])
        except:
            w.append(0)
    model['w']=csr_matrix(w)
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


def simple_hcca3(csr_mat, labels, model, step=10 ,initial=200, pos_limit=5, thres=30, stop=0.9, clustering=False, sample=2):
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

        if pos >= total: ## stop rule
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

        if idx<sample and clustering:
            can=init_sample(csr_mat,step,sample)
            train.extend(can)
            pool = list(set(pool) - set(can))
            try:
                pos = Counter(labels[train])["yes"]
            except:
                pos = 0
            pos_track.append(pos)
            continue

        order = np.argsort((model['w']*csr_mat[pool].transpose()).toarray()[0])
        if model['pos_at'] == 1:
            can=[pool[i] for i in order[-step:]]
        else:
            can=[pool[i] for i in order[:step]]
        # can=[pool[i] for i in order[:int(step/2)]]+[pool[i] for i in order[-step+int(step/2):]]
        train.extend(can)
        pool = list(set(pool) - set(can))
        try:
            pos = Counter(labels[train])["yes"]
        except:
            pos = 0
        pos_track.append(pos)

        if not begin:
            pool4 = pool[:]
            train4 = train[:]
            pos_track4 = pos_track[:]
            if pos>=pos_limit:
                begin=idx+1
        else:
            if round >= initial:
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
            else:
                can4 = np.random.choice(pool4, step, replace=False)
                train4.extend(can4)
                pool4 = list(set(pool4) - set(can4))
                pos = Counter(labels[train4])["yes"]
                pos_track4.append(pos)

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

## Start rule (clustering)
def init_sample(data,n_clusters,samples):
    cluster=KMeans(n_clusters=n_clusters)
    cluster.fit(data)
    result=cluster.labels_
    x=list(set(result))
    pool=[]
    for key in x:
        a=[i for i in xrange(data.shape[0])if result[i]==key]
        pool.extend(list(np.random.choice(a,samples,replace=False)))
    return pool

if __name__ == "__main__":
    eval(cmd())
