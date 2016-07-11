from __future__ import division, print_function


from pdb import set_trace

# from subprocess import Popen, CREATE_NEW_CONSOLE
from injest import xml2elastic, defaults


# from funcs import *


class ESHandler:
    def __init__(self, es=None, force_injest=False):
        self.es = es if es else defaults(TYPE_NAME='acm')
        self.injest(force=force_injest)
        self.target="shriram krishnamurthi"

    def __status_check_(self):
        self.es.INDEX_NAMEed = self.es.ES_CLIENT.indices.exists(
                index=self.es.INDEX_NAME)

        self.mapped = self.es.ES_CLIENT.indices.exists_type(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME)

        self.ready = self.es.INDEX_NAMEed and self.mapped

    def injest(self, force=False):

        try:
            self.__status_check_()
        except self.es.ES_EXCEPTIONS.ConnectionError:
            print("ES Server not started. Now starting ... ")
            # Popen('elasticsearch.bat', creationflags=CREATE_NEW_CONSOLE)
            self.__status_check_()


        if force:
            print(
                    'ESHandler: Database not ready (or) Force injest '
                    'requested. Now indexing...')
            dir="../data/citeseerx/citemap.csv"
            xml2es = xml2elastic(renew=True,verbose=True)
            self.es = xml2es.parse(dir)

    def query_string(self, keystring):
        DIS_MAX_QUERY ={
            "query": {
                "filtered": {
                    "query": {
                        "query_string": {
                            "fields": ["abstract._analyzed", "title._analyzed"],
                            "query": keystring,
                            "analyzer": "my_english",
                            "analyze_wildcard": "true"
                        }
                    },
                    "filter": {
                        "bool": {
                            "must": [{
                                "term": {
                                    "label": "none"
                                }
                            }, {
                                "term": {
                                    "is_control": "no"
                                }
                            }]
                        }
                    }
                }
            },
            "highlight" : {
                "pre_tags" : ["<span style=\"background-color: #aeeaae\">"],
                "post_tags" : ["</span>"],
                "fields" : {
                    "text._analyzed": {
                        "number_of_fragments": 0
                    }
                }
            }
        }

        num = self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=DIS_MAX_QUERY,
            size=0)["hits"]["total"]

        return self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=DIS_MAX_QUERY,
            size=num)

    def search(self, keyword):

        keyword = keyword.strip()
        THE_QUERY = {
            "query": {
                "multi_match": {
                    "query" : keyword,
                    "type"  : "most_fields",
                    "fields": ["abstract._analyzed", "title._analyzed"]
                }
            }
        }

        DIS_MAX_QUERY = {
            "query": {
                "filtered": {
                    "query" : {
                        "dis_max": {
                            "queries": [
                                {
                                    "match": {
                                        "title._analyzed": keyword
                                    }
                                },
                                {
                                    "match": {
                                        "abstract._analyzed": keyword
                                    }
                                }
                            ]
                        }
                    },
                    "filter": {
                        "bool": {
                            "must": [{
                                "term": {
                                    "label": "none"
                                }
                            }, {
                                "term": {
                                    "is_control": "no"
                                }
                            }]
                        }
                    }
                }
            }
        }

        num = self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                body=DIS_MAX_QUERY,
                size=0)["hits"]["total"]

        return self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                body=DIS_MAX_QUERY,
                size=num)

    def get_labeled(self, labeled_later=False):
        QUERY = {
            "query": {
                "bool": {
                    "must"    : {
                        "term": {
                            "is_control": "no"
                        }
                    },
                    "must_not": {
                        "term": {
                            "label": "none"
                        }
                    },
                }
            }
        }

        "Note to self: Implement this with scroll"
        res = self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                body=QUERY,
                size=0)

        num = res["hits"]["total"]
        print(num)
        return self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                body=QUERY,
                size=num)

    def user_labeled(self):
        QUERY = {
            "query": {
                "bool": {
                    "must"    : {
                        "term": {
                            "is_control": "no"
                        }
                    },
                    "must_not": {
                        "term": {
                            "user": "no"
                        }
                    },
                }
            }
        }

        "Note to self: Implement this with scroll"
        res = self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                body=QUERY,
                size=0)

        num = res["hits"]["total"]
        print(num)
        return self.es.ES_CLIENT.search(
                index="citeseer",
                doc_type=self.es.TYPE_NAME,
                body=QUERY,
                size=num)

    def get_unlabeled(self):
        QUERY = {
            "query": {
                "bool": {
                    "must": [{
                        "term": {
                            "is_control": "no"
                        }
                    },{
                        "term": {
                            "label": "none"
                        }
                    }]
                }
            }
        }
        num = self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                body=QUERY,
                size=0)["hits"]["total"]

        return self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                body=QUERY,
                size=num)

    def get_specific(self, label, must=True, field='tags', total=500):
        state = "must" if must else "must_not"
        QUERY = {
            "query": {
                "bool": {
                    "must": {"match": {"label": "none"}},
                    state : {"match": {field: label}}
                }
            }
        }

        return self.es.ES_CLIENT.search(
                index=self.es.INDEX_NAME,
                body=QUERY,
                size=total)

    def set_label(self, _id, label):
        UPDATE = {
            "doc": {
                "label": str(label),
                "user" : "yes"
            }
        }
        self.es.ES_CLIENT.update(index=self.es.INDEX_NAME,
                                 doc_type=self.es.TYPE_NAME, id=_id,
                                 body=UPDATE)
        self.es.ES_CLIENT.indices.refresh(index=self.es.INDEX_NAME)

    def set_control(self, _id, sign="yes"):
        UPDATE = {
            "doc": {
                "is_control": sign
            }
        }
        self.es.ES_CLIENT.update(index=self.es.INDEX_NAME,
                                 doc_type=self.es.TYPE_NAME, id=_id,
                                 body=UPDATE)
        self.es.ES_CLIENT.indices.refresh(index=self.es.INDEX_NAME)

    def set_user(self, _id, sign="yes"):
        UPDATE = {
            "doc": {
                "user": sign
            }
        }
        self.es.ES_CLIENT.update(index=self.es.INDEX_NAME,
                                 doc_type=self.es.TYPE_NAME, id=_id,
                                 body=UPDATE)
        self.es.ES_CLIENT.indices.refresh(index=self.es.INDEX_NAME)

    def match_title(self,title):
        QUERY = {
            "query": {
                "filtered": {
                    "query": {
                        "match_phrase": {
                            "title._analyzed": title
                        }
                    },
                    "filter": {
                        "term": {
                            "is_control": "yes"
                        }
                    }
                }

            }
        }
        res = self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=QUERY,
            size=1)

        return res

    def get_control(self):
        QUERY = {
            "query": {
                "bool": {
                    "must": {
                        "term": {
                            "is_control": "yes"
                        }
                    }
                }
            }
        }

        res = self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=QUERY,
            size=0)

        num = res["hits"]["total"]
        print(num)
        return self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=QUERY,
            size=num)

    def get_target(self):
        QUERY = {
            "query": {
                "bool": {
                    "must": [{
                        "term": {
                            "is_control": "no"
                        }
                    }, {
                        "match": {
                            "tags": "pronunciation"
                        }
                    }]
                }
            }
        }
        res = self.es.ES_CLIENT.search(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=QUERY,
            size=0)
        return res['hits']['total']

    def get_document(self, _id):
        return self.es.ES_CLIENT.get(index=self.es.INDEX_NAME,
                                     doc_type=self.es.TYPE_NAME, id=_id)

    def reset_labels(self):
        res = self.get_labeled()['hits']['hits']
        for which in res:
            self.set_label(_id=which['_id'], label="none")
        control = self.get_control()['hits']['hits']
        for which in control:
            self.set_label(_id=which['_id'], label= "pos" if self.target in which['_source']['tags'] else "neg")

