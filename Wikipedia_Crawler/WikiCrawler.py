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
starting_page = "https://en.wikipedia.org/wiki/Spacetime"
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

#Extract the title tag
def extract_title(page):
    start_title = page.find("<span dir")
    end_start_title = page.find(">",start_title+1)
    stop_title = page.find("</span>", end_start_title + 1)
    print(start_title,end_start_title,stop_title)
    title = page[end_start_title + 1 : stop_title]
    return title

links=[]
crawled=[seed_page]
Links_Base[seed_page]+=1
def get_page(page=starting_page):
    global links,Knowledge_Base
    resp=download_page(page)
    resp=re.sub(r'<script>.+?</script>','',resp)
    breaks=resp.split('\\n')
    breaks=re.findall(r'<a href=(.+?)/a>',resp)
    for break_ in breaks:
        if('https' in break_ and 'wikipedia' in break_):
           link=re.findall('"(.+?)"',break_)
           if(link[0] not in crawled and link[0] not in links):
              links.append(link[0])

        title=re.findall(r'title=(.+?)>.+?<',break_)
        for t in title:
            t=re.sub(r'[-\\().\'_]',"",t)
            if(re.findall(r'ampaction|[0-9]|[:=&]', t)):
               pass
            else:
               print("********************",t,Knowledge_Base[t])
               Knowledge_Base[t]+=1

def url_parse(url):
    a = ['.png','.jpg','.jpeg','.gif','.tif','.txt']
    if(any(ext in url for ext in a)):
       return False
    else:
       return True

get_page(seed_page)
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
   
    if(len(Links_Base.keys())==5000):
       break

pickle.dump(Knowledge_Base,open('KnowledgeBase.pkl','wb'))
pickle.dump(Links_Base,open('LinksBase.pkl','wb'))

