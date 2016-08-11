"""
Create Term Frequency Matrix from Elastic Search Index.
"""

from __future__ import division, print_function

import os
import sys

root = os.getcwd().split("src")[0] + "src"
sys.path.append(root)

from pdb import set_trace
from injest import xml2elastic, defaults
import cPickle
import numpy as np
from scipy.sparse import csr_matrix
from ES_CORE import ESHandler
# from subprocess import Popen, CREATE_NEW_CONSOLE


class Pickle:
    def __init__(self, site='english', fname='vocabulary', verbose=False):
        self.site = site
        self.fname = fname
        self.verbose = verbose

    def load(self):
        if self.verbose:
            print("Loading: {filename}".format(filename=self.fname))

        with open(
                os.path.abspath("../dump/{filename}.pkl".format(
                        root=root,
                        site=self.site,
                        filename=self.fname)), "rb") as f:
            return cPickle.load(f)

    def brew(self, dat):
        try:
            cPickle.dump(dat, open(os.path.abspath(
                    "../dump/{filename}.pkl".format(
                            root=root,
                            site=self.site,
                            filename=self.fname)), "wb"))
            if self.verbose:
                print("Pickled {file}".format(file=self.fname))
            return True
        except Exception, e:
            raise e


class TermFrequency:
    def __init__(self, site="english", force_injest=False, verbose=False):
        # New elasticsearch wrapper
        self.exceptions = dict()
        self.force_injest = force_injest
        self.verbose = verbose
        self.es = defaults(TYPE_NAME=site)
        self.field_analyze="abstract._analyzed"

    def __status_check_(self):
        try:
            indexed = self.es.ES_CLIENT.indices.exists(
                    index=self.es.INDEX_NAME)

            mapped = self.es.ES_CLIENT.indices.exists_type(
                    index=self.es.INDEX_NAME,
                    doc_type=self.es.TYPE_NAME)

            if self.verbose == True:
                if indexed:
                    print("\tCheck 1/2: Dataset indexed.")
                if mapped:
                    print("\tCheck 2/2: Dataset mapped.")

            self.ready = indexed and mapped
            return True

        except self.es.ES_EXCEPTIONS.ConnectionError:
            print("Starting ES Server ... ")
            # Popen('elasticsearch.bat', creationflags=CREATE_NEW_CONSOLE)
            return False

    def injest(self):

        while not self.__status_check_():
            pass

        def find(lst, str):
            for elm in lst:
                if str in elm:
                    return elm
            return None

        try:
            if not self.ready or self.force_injest:
                self.force_injest = False
                if self.verbose:
                    print(
                            'TermFrequency: Database not ready (or) Force '
                            'injest '
                            'requested. Now indexing...')
                datasets = []
                for (dirpath, _, _) in os.walk("{dir}/data/".format(dir=root)):
                    datasets.append(os.path.abspath(dirpath))
                # discard the "../" dir
                datasets.pop(0)
                xml2es = xml2elastic(renew=True, verbose=self.verbose)
                self.es = xml2es.parse(find(datasets, str=self.es.TYPE_NAME))
                return True
            else:
                if self.verbose:
                    print("Site already injested in Elasticsearch.")

        except Exception, e:
            self.exceptions.update({'injest': e})
            return False

    def getVocab(self):
        ""
        self.injest()
        pkl = Pickle(site=self.es.TYPE_NAME, fname='vocabulary')

        try:
            return pkl.load()
        except:
            VOCAB = dict()
            TERM_INDX = 0
            for _, mapping in self.scroll(all=True, control=False):
                for term in mapping.keys():
                    if not term in VOCAB.keys():
                        VOCAB.update({term: TERM_INDX})
                        TERM_INDX += 1
            pkl.brew(dat=VOCAB)
            return pkl.load()

            # if self.isPickled:
            # else:
            #     return VOCAB

    def getTermVector(self, doc_id):
        mapping = dict()
        # --- Only analyzing body, need to include title ---
        post = self.es.ES_CLIENT.termvectors(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                id=doc_id,
                field_statistics=True,
                fields=[self.field_analyze],
                term_statistics=True)
        try:
            termVect_now = post[
                "term_vectors"][
                self.field_analyze][
                "terms"]
            tokens = termVect_now.keys()
            for token in tokens:
                mapping.update({token: {"tf": termVect_now[token]["term_freq"]}})
        except:
            pass


        return post["_id"], mapping

    def scroll(self, control, labeled=None, all=True):

        Q_ALL = {
            "query": {
                "bool": {
                    "must": {
                        "match": {"is_control": "yes" if control else "no"}
                    }
                }
            }
        }

        Q_LABELED = {
            "query": {
                "bool": {
                    "must"    : {
                        "match": {"is_control": "yes" if control else "no"}
                    },
                    "must_not": {"match": {"label": "none"}}
                }
            }
        }

        Q_UNLABELED = {
            "query": {
                "bool": {
                    "must": {
                        "match": {"is_control": "yes" if control else "no"}
                    },
                    "must": {"match": {"label": "none"}}
                }
            }
        }

        res = self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type = self.es.TYPE_NAME,
                scroll="1m",
                size=10,
                body=Q_ALL if all else Q_LABELED if labeled else Q_UNLABELED)

        hits = res["hits"]["hits"]
        for doc in hits:
            yield self.getTermVector(doc_id=doc["_id"])

        remaining = res["hits"]["total"]

        while remaining:
            my_scroll_id = res["_scroll_id"]
            res = self.es.ES_CLIENT.scroll(
                    scroll_id=my_scroll_id,
                    scroll="1m")
            hits = res["hits"]["hits"]
            for doc in hits:
                yield self.getTermVector(doc_id=doc["_id"])
            remaining = len(res["hits"]["hits"])

    def toCSR(self, term_map, control, labeled=None):
        ROW_OFFSET = [0]
        COL_INDICES = []
        VALUES = []
        DOC_ID = []
        ALL=False
        if labeled==None:
            ALL=True
        for doc, vector in self.scroll(control, labeled, all=ALL):
            prev = ROW_OFFSET[-1]
            ROW_OFFSET.append(prev + len(vector))
            DOC_ID.append(doc)
            # set_trace()
            for term, count in vector.iteritems():
                try:
                    val = term_map[term]
                except KeyError:
                    val = 0

                COL_INDICES.append(val)
                VALUES.append(count["tf"])
        return np.asarray(ROW_OFFSET), np.asarray(COL_INDICES), np.asarray(
                VALUES), np.asarray(DOC_ID)

    def matrix(self, CONTROL=False, LABELED=None):
        term_map = self.getVocab()
        indptr, indices, data, label_id = self.toCSR(term_map, control=CONTROL,
                                                     labeled=LABELED)
        N_rows, N_cols = len(label_id), len(term_map.keys())
        MY_ES_DSL = ESHandler(es=self.es, force_injest=False)
        return {
            "header": sorted(term_map.keys(), key=lambda F: term_map[F]),
            "matrix": csr_matrix((data, indices, indptr),
                                 shape=(N_rows, N_cols)),
            "meta"  : [{
                           "doc_id": _id,
                           "user": MY_ES_DSL.get_document(_id)["_source"][
                                   "user"],
                           "label" :
                               MY_ES_DSL.get_document(_id)["_source"][
                                   "label"]
                       } for _id in label_id]
        }
