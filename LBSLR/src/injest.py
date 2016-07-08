from __future__ import print_function, division
from elasticsearch import Elasticsearch, exceptions

import unicodedata
from tika import parser
from pdb import set_trace

class Vessel(object):
    id = -1

    def __init__(i, **fields):
        i.override(fields)
        i.newId()

    def newId(i):
        i._id = Vessel.id = Vessel.id + 1

    def also(i, **d):
        return i.override(d)

    def override(i, d):
        i.__dict__.update(d)
        return i

    def __hash__(i):
        return i._id

def defaults(**d):
    """Deafult ssetting to enable ES index"""

    The = Vessel(
        ES_HOST={
            "host": 'localhost',
            "port": 9200},
        INDEX_NAME="citeseer",
        TYPE_NAME="",
        ANALYZER_NAME="my_english",
        ANALYZER_NAME_SHINGLE="my_english_shingle")

    The.also(ES_CLIENT=Elasticsearch(
        hosts=[The.ES_HOST],
        timeout=10,
        max_retries=10,
        retry_on_timeout=True))

    The.also(ES_CLIENT_ORIG=Elasticsearch(
        hosts=[The.ES_HOST],
        timeout=10,
        max_retries=10,
        retry_on_timeout=True))

    The.also(ES_EXCEPTIONS=exceptions)

    if d:
        The.override(d)

    return The

class xml2elastic:
    def __init__(self, renew=True, verbose=False):
        self.es = defaults()
        self.renew = renew
        self.verbose = verbose
        self.init_index()

    def init_index(self):

        if self.renew:
            self.es.ES_CLIENT.indices.delete(
                index=self.es.INDEX_NAME,
                ignore=[400, 404])

        try:
            self.es.ES_CLIENT.indices.create(
                index=self.es.INDEX_NAME,
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                self.es.ANALYZER_NAME: {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "char_filter": [
                                        "html_strip"
                                    ],
                                    "filter": [
                                        "lowercase",
                                        "asciifolding",
                                        "stopper",
                                        "my_snow"
                                    ]
                                }
                            },
                            "filter": {
                                "stopper": {
                                    "type": "stop",
                                    "stopwords": "_english_"
                                },
                                "my_snow": {
                                    "type": "snowball",
                                    "language": "English"
                                }
                            }
                        }
                    }
                }
            )
            if self.verbose: print('Step 1 of 3: Indices Created.')
        except Exception, e:
            print(e)
            set_trace()
            if self.verbose: print('Step 1 of 3: Indices already exist.')

    def init_mapping(self, doc_type=None):
        self.es.also(TYPE_NAME=doc_type)
        mapping = {
            self.es.TYPE_NAME: {
                "properties": {
                    "title": {
                        "type": "multi_field",
                        "fields": {
                            "title": {
                                "include_in_all": True,
                                "type": "string",
                                "store": True,
                                "index": "not_analyzed"
                            },
                            "_analyzed": {
                                "type": "string",
                                "store": True,
                                "index": "analyzed",
                                "term_vector": "with_positions_offsets",
                                "analyzer": self.es.ANALYZER_NAME
                            }
                        }
                    },
                    "abstract": {
                        "type": "multi_field",
                        "fields": {
                            "abstract": {
                                "include_in_all": True,
                                "type": "string",
                                "store": True,
                                "index": "not_analyzed"
                            },
                            "_analyzed": {
                                "type": "string",
                                "store": True,
                                "index": "analyzed",
                                "term_vector": "with_positions_offsets",
                                "analyzer": self.es.ANALYZER_NAME
                            }
                        }
                    },
                    "year": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "authors": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "conference": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "label": {
                        "type": "string",
                        "index": "not_analyzed"
                    }
                }
            }
        }

        self.es.ES_CLIENT.indices.put_mapping(
            index=self.es.INDEX_NAME,
            doc_type=self.es.TYPE_NAME,
            body=mapping)
        if self.verbose: print('Step 2 of 3: Docment mapped.')

    @staticmethod
    def decode(dir):
        dir = '../data/citeseerx/citemap.csv'
        with open(dir,'rb') as f:
            spamreader=f.readlines()
            for idx, row in enumerate(spamreader[1:]):
                row=row.strip().split("$|$")
                conference = str(row[1].encode('ascii', 'ignore'))
                year = str(row[2].encode('ascii', 'ignore'))
                title = str( unicode(row[3],errors="ignore").encode('ascii', 'ignore'))
                abstract = str(unicode(row[-1],errors="ignore").encode('ascii', 'ignore'))
                authors = row[-2].encode('ascii', 'ignore').split(',')
                yield idx, conference, year, title, abstract, filter(None, authors)

    @staticmethod
    def decode_acm():
        dir = '../data/five/acm.csv'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for idx, row in enumerate(spamreader[1:]):
                row = row.strip().split("$|$")
                title = str(row[0].encode('ascii', 'ignore'))
                authors = row[1].encode('ascii', 'ignore').split(',')
                year = str(row[2].encode('ascii', 'ignore'))
                citedCount = str(row[3].encode('ascii', 'ignore'))
                ft_url = row[3]
                if not ft_url:
                    continue
                ft = unicodedata.normalize('NFKD', parser.from_file(ft_url)["content"].strip()).encode('ascii', 'ignore')

                yield idx, title, filter(None, authors), year, citedCount, ft

    @staticmethod
    def decode_ieee():
        dir = '../data/five/ieee.csv'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for idx, row in enumerate(spamreader[2:]):
                row = row.strip().split("\",\"")
                title = str(row[0].encode('ascii', 'ignore'))
                authors = row[1].encode('ascii', 'ignore').split(';')
                year = str(row[5].encode('ascii', 'ignore'))
                citedCount = str(row[21].encode('ascii', 'ignore'))
                ft_url = row[15]
                if not ft_url:
                    continue
                ft = unicodedata.normalize('NFKD', parser.from_file(ft_url)["content"].strip()).encode('ascii',
                                                                                                       'ignore')

                yield idx, title, filter(None, authors), year, citedCount, ft



    def parse(self, dir, fresh=True):

        target="shriram krishnamurthi"
        "Parse XML to ES Database"
        if self.verbose: print("Injesting: {}\r".format(dir), end='\n')

        # Create Mapping
        self.init_mapping(doc_type="citemap")
        MAX_RELEVANT = 250
        MAX_IRRELEVANT = 250
        MAX_CONTROL = 1500

        for (idx, title, authors, year, citedCount, ft_url) in self.decode_acm():
            # CONTROL = True if random() < 0.1 and MAX_CONTROL > 0 else False
            # if CONTROL: MAX_CONTROL -= 1
            CONTROL = False
            REAL_TAG = 'pos' if target in authors else 'neg'
            content = {
                "citedCount": citedCount,
                "year": year,
                "title": title,
                "ft_url":  ft_url,
                "authors":  authors,
                "label": REAL_TAG  if CONTROL else 'none',
                "is_control":"yes" if CONTROL else "no",
                "user": "no"
            }
            self.es.ES_CLIENT.index(
                index='citeseer',
                doc_type="citemap",
                id=idx,
                body=content)

            self.es.ES_CLIENT_ORIG.index(
                index='citeseer',
                doc_type="citemap",
                id=idx,
                body=content)

            if self.verbose:
                print("Post #{id} injested\r".format(id=idx), end="")


        if self.verbose: print('Step 3 of 3: Site injested. Total Documents injested: {}.\n'.format(idx))
        return self.es
