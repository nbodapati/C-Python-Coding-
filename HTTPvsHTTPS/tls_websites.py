import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
from PIL import Image
from io import BytesIO
import pickle
import subprocess 
from collections import defaultdict
import pandas as pd

# This is the 2.11 Requests cipher string, containing 3DES.
CIPHERS = (
    'ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:ECDH+HIGH:'
    'DH+HIGH:ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+HIGH:RSA+3DES:!aNULL:'
    '!eNULL:!MD5''ECDHE+ECDSA+AES256'
)

top_1m_sites=pd.read_csv('./top-1m.csv',header=None)
websites=top_1m_sites.iloc[:,1]
suite_dict=defaultdict(int)
TLSv2_websites=[]

def print_dict(d):
    for k,v in d.items():
        print(k,":",v)

def dump_dict(d): 
    pickle.dump(d,open('suite_dict.pkl','wb'))    

def process_output(op):
    import re
    op=re.sub(r'[\s]+','',op)
    op=re.sub(r'[|-]',',',op)
    op= op.strip(':|-_\n\t')
    op=op.split(',')
    try:
      if(op[1][:3]=='TLS'):
         return op[1]
    except:
         return None

def find_index(lst,website):
    global suite_dict
    str='TLSv1.2'
    try:
      idx=lst.index(str)
      with open('TLSv1.2_websites2.csv','a') as fp:
           fp.write("%s\n"%(website))     
           fp.close()    
    except:
      return 

    z=lst[idx:]
    for l in z:
        suite_dict[l]+=1    
    #suite_dict=dict(zip(z,[1]*len(z)))
    #print_dict(suite_dict)
    dump_dict(suite_dict)
    

def reachable_by_http(website):
    http_req="http://www." + website  
    #print("Accessing: ",http_req)
    try:
      r=requests.get(http_req,timeout=1)
      return True  
    except requests.exceptions.SSLError:
      #print("Not reachable to http: ",http_req)    
      return False   
    except:
      #print("Error found: ",sys.exc_info()[0])
      return False   
    if(r.history): 
      #it got redirected to https --so cant be accessed.
      #print("Requested:",http_req," Redirected_to:",r.url)
      return False
 
def reachable_by_https(website):
    https_req="https://www." + website  
    #print("Accessing: ",https_req)
    try:
      r=requests.get(https_req,timeout=1)
      return True  
    except requests.exceptions.SSLError:
      #print("Not reachable to http: ",http_req)    
      return False   
    except:
      #print("Error found: ",sys.exc_info()[0])
      return False   
    if(r.history): 
      #it got redirected to https --so cant be accessed.
      #print("Requested:",http_req," Redirected_to:",r.url)
      return False


if __name__ =='__main__':
   import sys
   print("Starting the code...")
   for i,website in enumerate(websites):
       sys.stdout.write("\rAccessing website number %i" % i)
       sys.stdout.flush() 
       #check if this supports http also.
       if(reachable_by_http(website)==False):
          continue 
       #check if this supports https also.
       if(reachable_by_https(website)==False):
          continue 
 
       print(website)
       reqstr='nmap --script ssl-enum-ciphers -p 443 '+website
       p=subprocess.Popen(reqstr, shell=True,\
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
       lines=[]
       for line in p.stdout.readlines()[7:]:
           op=process_output(line)
           if(op!=None):
              lines.append(op)
       find_index(lines,website)



