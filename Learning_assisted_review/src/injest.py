from __future__ import print_function, division
from elasticsearch import Elasticsearch, exceptions

import unicodedata
from tika import parser
from pdb import set_trace
import urllib2
import time

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
        INDEX_NAME="slr",
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
                    "doi": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "ft_url": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "citedCount": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "label": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "user": {
                        "type": "string",
                        "index": "not_analyzed"
                    },
                    "is_control": {
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
        temp_pdf="../temp/temp.pdf"
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for idx,row in enumerate(spamreader[1:]):
                row = row.strip().split("$|$")
                title = str(row[0].encode('ascii', 'ignore'))
                authors = row[1].encode('ascii', 'ignore').split(',')
                year = str(row[2].encode('ascii', 'ignore'))
                citedCount = str(row[3].encode('ascii', 'ignore'))
                ft_url = row[4]

                if not ft_url:
                    continue
                try:
                    req = urllib2.Request(ft_url, headers={'User-Agent': "Magic Browser"})
                    con = urllib2.urlopen(req)
                    page = con.read()
                    con.close()
                    time.sleep(10)
                except:
                    print("Stop at: %d" %idx)
                    exit()

                with open(temp_pdf,'w') as f:
                    f.write(page)
                ft = unicodedata.normalize('NFKD', parser.from_file(temp_pdf)["content"].strip()).encode('ascii', 'ignore')
                yield title, filter(None, authors), year, citedCount, ft_url, ft

    @staticmethod
    def decode_Hall():
        dir = '../data/Hall/Hall.txt'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for row in spamreader[1:]:
                row = row.strip().split("\t")
                # title = unicodedata.normalize('NFKD', row[0].strip()).encode('ascii', 'ignore')
                # authors = unicodedata.normalize('NFKD', row[1].strip()).encode('ascii', 'ignore').split(';')
                title = str(row[0].strip())
                if not title:
                    continue
                authors = row[1].split('; ')
                year = str(row[5])
                citedCount = str(row[21])
                doi = row[14].strip().lower()
                ft_url = row[15]
                abstract = row[10].strip()

                yield title, filter(None, authors), year, citedCount, ft_url, abstract , doi

    @staticmethod
    def decode_ieee():
        dir = '../data/five/ieee.csv'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for row in spamreader[1:]:
                row = row.strip()[1:-1].split("\",\"")
                # title = unicodedata.normalize('NFKD', row[0].strip()).encode('ascii', 'ignore')
                # authors = unicodedata.normalize('NFKD', row[1].strip()).encode('ascii', 'ignore').split(';')
                title = str(row[0].strip())
                if not title:
                    continue
                authors = row[1].split('; ')
                year = str(row[5])
                citedCount = str(row[21])
                doi = row[14].strip().lower()
                ft_url = row[15]
                abstract = row[10].strip()

                yield title, filter(None, authors), year, citedCount, ft_url, abstract, doi

    @staticmethod
    def decode_final_list():
        dir = '../data/final_list/final_list.csv'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for row in spamreader[0:]:
                row = row.strip().split("$|$")
                # title = unicodedata.normalize('NFKD', row[0].strip()).encode('ascii', 'ignore')
                # authors = unicodedata.normalize('NFKD', row[1].strip()).encode('ascii', 'ignore').split(';')
                source = row[0].strip()
                if not (source=="ieee" or source=="acm"):
                    continue
                title = str(row[1].strip())
                doi = row[2].strip().lower()
                citedCount = str(row[3].strip())
                abstract = str(row[4].strip())

                yield title, doi, citedCount, abstract

    @staticmethod
    def decode_contain():
        dir = '../data/Hall/contain.txt'
        with open(dir, 'rb') as f:
            spamreader = f.readlines()
            for row in spamreader:
                doi = row.strip().lower()
                yield doi



    def parse_Hall(self, dir, fresh=True):

        "Parse XML to ES Database"
        if self.verbose: print("Injesting: {}\r".format(dir), end='\n')

        # Create Mapping
        self.init_mapping(doc_type="Hall")
        MAX_RELEVANT = 250
        MAX_IRRELEVANT = 250
        MAX_CONTROL = 1500

        for idx, (title, authors, year, citedCount, ft_url, abstract, doi) in enumerate(self.decode_Hall()):
            # CONTROL = True if random() < 0.1 and MAX_CONTROL > 0 else False
            # if CONTROL: MAX_CONTROL -= 1
            CONTROL = False
            content = {
                "citedCount": citedCount,
                "year": year,
                "title": title,
                "ft_url":  ft_url,
                "abstract":  abstract,
                "authors":  authors,
                "doi":  doi,
                "label": 'none',
                "is_control":"yes" if CONTROL else "no",
                "user": "no"
            }
            self.es.ES_CLIENT.index(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                id=idx,
                body=content)

            self.es.ES_CLIENT_ORIG.index(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                id=idx,
                body=content)

            if self.verbose:
                print("Post #{id} injested\r".format(id=idx), end="")

        total=idx

        for doi in self.decode_contain():
            BODY = {
                "query": {
                    "term": {
                        "doi": doi
                    }
                }
            }
            req = self.es.ES_CLIENT.search(index=self.es.INDEX_NAME,
                                           doc_type=self.es.TYPE_NAME, body=BODY, size=1)
            if req["hits"]["total"] > 0:
                UPDATE = {
                    "doc": {
                        "user": "yes"
                    }
                }
                self.es.ES_CLIENT.update(index=self.es.INDEX_NAME,
                                         doc_type=self.es.TYPE_NAME, id=req["hits"]["hits"][0]["_id"],
                                         body=UPDATE)



        self.es.ES_CLIENT.indices.refresh(index=self.es.INDEX_NAME)



        if self.verbose: print('Step 3 of 3: Site injested. Total Documents injested: {}.\n'.format(idx+1))
        return self.es

    def parse_ieee(self, dir, fresh=True):

        "Parse XML to ES Database"
        if self.verbose: print("Injesting: {}\r".format(dir), end='\n')

        # Create Mapping
        self.init_mapping(doc_type="ieee")
        MAX_RELEVANT = 250
        MAX_IRRELEVANT = 250
        MAX_CONTROL = 1500

        for idx, (title, authors, year, citedCount, ft_url, abstract, doi) in enumerate(self.decode_ieee()):
            # CONTROL = True if random() < 0.1 and MAX_CONTROL > 0 else False
            # if CONTROL: MAX_CONTROL -= 1
            CONTROL = False
            content = {
                "citedCount": citedCount,
                "year": year,
                "title": title,
                "ft_url": ft_url,
                "abstract": abstract,
                "authors": authors,
                "doi": doi,
                "label": 'none',
                "is_control": "yes" if CONTROL else "no",
                "user": "no"
            }
            self.es.ES_CLIENT.index(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                id=idx,
                body=content)

            self.es.ES_CLIENT_ORIG.index(
                index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,
                id=idx,
                body=content)

            if self.verbose:
                print("Post #{id} injested\r".format(id=idx), end="")

        total = idx



        for (title, doi, citedCount, abstract) in self.decode_final_list():

            BODY={
                "query": {
                    "term": {
                        "doi": doi
                    }
                }
            }
            req=self.es.ES_CLIENT.search(index=self.es.INDEX_NAME,
                doc_type=self.es.TYPE_NAME,body=BODY,size=1)
            if req["hits"]["total"]>0:
                UPDATE = {
                    "doc": {
                        "user": "yes"
                    }
                }
                self.es.ES_CLIENT.update(index=self.es.INDEX_NAME,
                                         doc_type=self.es.TYPE_NAME, id=req["hits"]["hits"][0]["_id"],
                                         body=UPDATE)
            else:
                idx=idx+1
                content = {
                    "citedCount": citedCount,
                    "year": "",
                    "title": title,
                    "ft_url": "",
                    "abstract": abstract,
                    "authors": [],
                    "doi": doi,
                    "label": 'none',
                    "is_control": "no",
                    "user": "yes"
                }
                self.es.ES_CLIENT.index(
                    index=self.es.INDEX_NAME,
                    doc_type=self.es.TYPE_NAME,
                    id=idx,
                    body=content)

        self.es.ES_CLIENT.indices.refresh(index=self.es.INDEX_NAME)
        print("new: %d" %(idx-total))


        if self.verbose: print('Step 3 of 3: Site injested. Total Documents injested: {}.\n'.format(idx + 1))
        return self.es
