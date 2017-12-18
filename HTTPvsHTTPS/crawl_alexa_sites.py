#Code for website crawling to check which of the sites can be accessible by both http 
#and https

import numpy as np
import pandas as pd
import requests 
import json
import sys
import time
top_1m_sites=pd.read_csv('./top-1m.csv',header=None)
#print(top_1m_sites.columns,top_1m_sites.head(),top_1m_sites.shape)

websites=top_1m_sites.iloc[:,1]

only_http=[]
both_http_https=[]
only_https=[]

ssl_error_http=[]
ssl_error_https=[]


for website in websites:
    http_req="http://www." + website  
    print("Accessing: ",http_req)
    try:
      start=time.time();r=requests.get(http_req,timeout=1);http_time=(time.time()-start);
    except requests.exceptions.SSLError:
      print("Not reachable to http: ",http_req)       
      ssl_error_http.append(http_req)
      continue
    except:
      print("Error found: ",sys.exc_info()[0])
      continue
   
    if(r.history): 
      #it got redirected to https --so cant be accessed.
      print("Requested:",http_req," Redirected_to:",r.url)
      only_https.append(http_req)
    else: 
      https_req="https://www." + website  
      print("Accessing: ",https_req)
      try:
         start=time.time();rs=requests.get(https_req,timeout=1);https_time=(time.time()-start);
      except requests.exceptions.SSLError:
         print("Not reachable to https: ",https_req)       
         ssl_error_https.append(https_req)
         continue
      except:
         #print("Error found: ",sys.exc_info()[0])
         continue

      if(rs.history): 
         #it got redirected to some other --so cant be accessed.
         print("Requested:",https_req," Redirected_to:",rs.url)
         only_http.append(http_req)
      else:
         both_http_https.append(http_req) 
         print("%s,%0.4f,%0.4f\n"%(website,http_time,https_time))  
         
         #when both are reachable, request each individually.
         http_times=[]
         https_times=[]
         for time_ in range(10):
             try:   
               start=time.time();r=requests.get(http_req);http_time=(time.time()-start);
               http_times.append(http_time)
             except:
               pass
             try:
               start=time.time();r=requests.get(https_req);https_time=(time.time()-start);
               https_times.append(https_time)
             except:
               pass
         if(len(http_times)!=0 and len(http_times)!=0):
            with open('crawl_both_reachable.csv','a') as fp:
                 fp.write("%s,%0.4f,%0.4f\n"%(website,sum(http_times)/len(http_times),\
                     sum(https_times)/len(https_times)))     
                 fp.close()    

with open('website_access.json','w') as fp:
     json.dump({'both_http_https':both_http_https,'only_http':only_http,'only_https':only_https},fp)

