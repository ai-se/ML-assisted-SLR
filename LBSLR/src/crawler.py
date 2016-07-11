from __future__ import print_function, division
from bs4 import BeautifulSoup
import urllib2
import unicodedata
from pdb import set_trace

def crawl_acm_old(url):
    den="$|$"
    # http="http://dl.acm.org.prox.lib.ncsu.edu/"
    http = "http://dl.acm.org/"
    csv_str="title"+den+"authors"+den+"year"+den+"citedCount" +den +"fulltext_url" + "\n"

    start=0
    while(True):
        if start:
            req = urllib2.Request(url+"&start="+str(start), headers={'User-Agent': "Magic Browser"})
        else:
            req = urllib2.Request(url, headers={'User-Agent': "Magic Browser"})

        page = urllib2.urlopen(req).read()

        soup=BeautifulSoup(page,'html.parser')

        title=soup.find_all('div','title')
        if not len(title):
            break
        authors=soup.find_all('div','authors')
        source=soup.find_all('div','source')
        metrics=soup.find_all('div','metrics')
        ft=soup.find_all('div','ft')

        start = start + len(title)
        # abstract = soup.find_all('div', 'abstract')
        # kw = soup.find_all('div', 'kw')
        for i in xrange(len(title)):
            csv_str=csv_str+title[i].find('a').get_text()+den
            try:
                for author in authors[i].find_all('a'):
                    csv_str = csv_str + unicodedata.normalize('NFKD', author.get_text()).encode('ascii','ignore') + ","
                csv_str=csv_str[:-1]+den
            except:
                csv_str=csv_str+den
            csv_str = csv_str + [x for x in source[i].find('span', 'publicationDate').get_text().split(' ') if x.isdigit()][-1] + den
            csv_str = csv_str + [x for x in metrics[i].find('span', 'citedCount').get_text().split(' ') if x.isdigit()][-1] + den
            csv_str = csv_str + http +ft[i].find('a').get('href').split('&')[0] + "\n"
    with open('../data/five/acm.csv', 'w') as f:
        f.write(csv_str)

def crawl_acm(url):
    den="$|$"
    # http="http://dl.acm.org.prox.lib.ncsu.edu/"
    http = "http://dl.acm.org/"
    csv_str="title"+den+"authors"+den+"year"+den+"citedCount" +den +"fulltext_url" + "\n"

    start=0
    while(True):
        if start:
            req = urllib2.Request(url+"&start="+str(start), headers={'User-Agent': "Magic Browser"})
        else:
            req = urllib2.Request(url, headers={'User-Agent': "Magic Browser"})

        page = urllib2.urlopen(req).read()

        soup=BeautifulSoup(page,'html.parser')

        title=soup.find_all('div','title')
        if not len(title):
            break

        start = start + len(title)

        current=-1
        should=-1

        for div in soup.find_all('div'):
            try:
                div_class=div['class']
            except:
                continue
            if div_class[0]=="title":
                should=should+1
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 - current - 1)
                if should * 5 - current:
                    csv_str = csv_str + "\n"
                current = should * 5
                csv_str = csv_str + unicodedata.normalize('NFKD', div.find('a').get_text()).encode('ascii', 'ignore') + den

            elif div_class[0]=="authors":
                current = current + 1
                csv_str=csv_str+"$|$"*(should* 5 + 1-current)
                current = should * 5 + 1
                try:
                    for author in div.find_all('a'):
                        csv_str = csv_str + unicodedata.normalize('NFKD', author.get_text()).encode('ascii', 'ignore') + ","
                    csv_str = csv_str[:-1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="source":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 2 - current)
                current = should * 5 + 2
                try:
                    csv_str = csv_str + \
                              [x for x in div.find('span', 'publicationDate').get_text().split(' ') if x.isdigit()][
                                  -1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="metrics":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 3 - current)
                current = should * 5 + 3
                try:
                    csv_str = csv_str + \
                              [x for x in div.find('span', 'citedCount').get_text().split(' ') if x.isdigit()][
                                  -1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="ft":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 4 - current)
                current = should * 5 + 4
                try:
                    csv_str = csv_str + http + div.find_all('a')[-1].get('href').split('&')[0] + "\n"
                except:
                    csv_str = csv_str + "\n"

        should = should + 1
        current = current + 1
        csv_str = csv_str + "$|$" * (should * 5 - 1 - current)
        if should * 5 - current:
            csv_str = csv_str + "\n"
        current = should * 5


    with open('../data/five/acm.csv', 'w') as f:
        f.write(csv_str)





def crawl_acm_doi(url):
    den="$|$"
    # http="http://dl.acm.org.prox.lib.ncsu.edu/"
    http = "http://dl.acm.org/"
    csv_str="title"+den+"authors"+den+"year"+den+"citedCount" +den +"fulltext_url" + den +"abstract" + den +"doi" + den+ "ISBN" + "\n"

    start=0
    while(True):
        if start:
            req = urllib2.Request(url+"&start="+str(start), headers={'User-Agent': "Magic Browser"})
        else:
            req = urllib2.Request(url, headers={'User-Agent': "Magic Browser"})

        page = urllib2.urlopen(req).read()

        soup=BeautifulSoup(page,'html.parser')

        title=soup.find_all('div','title')
        if not len(title):
            break

        start = start + len(title)

        current=-1
        should=-1

        for div in soup.find_all('div'):
            try:
                div_class=div['class']
            except:
                continue
            if div_class[0]=="title":
                should=should+1
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 - current - 1)
                if should * 5 - current:
                    csv_str = csv_str + "\n"
                current = should * 5
                csv_str = csv_str + unicodedata.normalize('NFKD', div.find('a').get_text()).encode('ascii', 'ignore') + den
                url2="http://dl.acm.org/" + div.find('a').get('href')+"&amp;preflayout=flat"
                req2 = urllib2.Request(url2, headers={'User-Agent': "Magic Browser"})
                con2 = urllib2.urlopen(req2)
                page2=con2.read()
                soup2=BeautifulSoup(page2,'html.parser')
                set_trace()

            elif div_class[0]=="authors":
                current = current + 1
                csv_str=csv_str+"$|$"*(should* 5 + 1-current)
                current = should * 5 + 1
                try:
                    for author in div.find_all('a'):
                        csv_str = csv_str + unicodedata.normalize('NFKD', author.get_text()).encode('ascii', 'ignore') + ","
                    csv_str = csv_str[:-1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="source":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 2 - current)
                current = should * 5 + 2
                try:
                    csv_str = csv_str + \
                              [x for x in div.find('span', 'publicationDate').get_text().split(' ') if x.isdigit()][
                                  -1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="metrics":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 3 - current)
                current = should * 5 + 3
                try:
                    csv_str = csv_str + \
                              [x for x in div.find('span', 'citedCount').get_text().split(' ') if x.isdigit()][
                                  -1] + den
                except:
                    csv_str = csv_str + den

            elif div_class[0]=="ft":
                current = current + 1
                csv_str = csv_str + "$|$" * (should * 5 + 4 - current)
                current = should * 5 + 4
                try:
                    csv_str = csv_str + http + div.find_all('a')[-1].get('href').split('&')[0] + "\n"
                except:
                    csv_str = csv_str + "\n"

        should = should + 1
        current = current + 1
        csv_str = csv_str + "$|$" * (should * 5 - 1 - current)
        if should * 5 - current:
            csv_str = csv_str + "\n"
        current = should * 5


    with open('../data/five/acm.csv', 'w') as f:
        f.write(csv_str)
