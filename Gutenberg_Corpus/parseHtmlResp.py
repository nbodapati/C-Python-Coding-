import urllib.request    
import re

#to write the content response from http request.
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

starting_page = "https://en.wikipedia.org/wiki/The_Scarlet_Letter"
resp=download_page(starting_page)

fd=open('links.txt','w')
resps=resp.split('\\n')
for r in resps:
    #print(r,end='\n') 
    print(r,end='\n',file=fd)

fd.close()

