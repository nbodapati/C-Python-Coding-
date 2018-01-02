#Import Libraries

import time    
import urllib.request    
import re
from collections import defaultdict
import pickle

#Global dictionary
Knowledge_Base=defaultdict(int)
Links_Base=defaultdict(int)

#Defining pages
starting_page = "https://en.wikipedia.org/wiki/The_Scarlet_Letter"
seed_page = "https://en.wikipedia.org" 

#Downloading entire Web Document (Raw Page Content)
def download_page(url):
    try:
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
        req = urllib.request.Request(url, headers = headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read())
        return respData
    except Exception as e:
        print(str(e))


links=[]
crawled=[starting_page]
Links_Base[starting_page]+=1
def get_page(page=starting_page):
    global links,Knowledge_Base
    resp=download_page(page)
    resps=re.findall(r'<a href=\"(.*?)\" title=.*?>(.*?)<',resp)   
    #list of tuples of ("html","keyphrase")
    for r_ in resps:
      if(r_[0].startswith('/wiki') and r_[1]!=''):
        r1=re.sub(r'[^A-Za-z0-9\s]','',r_[1])
        r0=r_[0].split('\" ')[0]
        print("********************",r1,Knowledge_Base[r1])
        link=seed_page+r0
        if(link not in crawled and link not in links):
              links.append(link)
        Knowledge_Base[r1]+=1
          

def url_parse(url):
    a = ['.png','.jpg','.jpeg','.gif','.tif','.txt']
    if(any(ext in url for ext in a)):
       return False
    else:
       return True

get_page(starting_page)
for link in links:
    print("**************Link: ",link,Links_Base[link])
    valid=url_parse(link)
    if(valid==False):
       print("*****************Invalid*************")  
       continue
    crawled.append(link)
    Links_Base[link]+=1
    print("Num links crawled:",len(crawled),len(Links_Base.keys()))
    print("Size of KB built:",len(Knowledge_Base.keys()))
    try:
      resp=get_page(link)
    except Exception as e:
      print("*********************Exception**************",str(e))
    
    if(len(Links_Base.keys())%1000==0):   
       pickle.dump(Knowledge_Base,open('KnowledgeBase.pkl','wb'))
       pickle.dump(Links_Base,open('LinksBase.pkl','wb'))
 
    if(len(Links_Base.keys())==50000):
       break

pickle.dump(Knowledge_Base,open('KnowledgeBase.pkl','wb'))
pickle.dump(Links_Base,open('LinksBase.pkl','wb'))

