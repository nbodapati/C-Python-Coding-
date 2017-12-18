from __future__ import division
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
import pickle
import subprocess 
from collections import defaultdict
import pandas as pd
import time

'''
ciphers=[('RSA+3DES+SHA'),('ECDHE+RSA+AES128+SHA256'),\
          ('ECDH+ECDSA+AESCBC'),\
          ('RSA+AES256+SHA256'),('RSA+AES256+SHA384'),\
          ('DHE+RSA+AES128+SHA'),('ECDHE+RSA+AES128+SHA'),\
          ('ECDHE+ECDSA+AES256+SHA384'),('ECDHE+RSA+3DES+SHA'),\
          ('ECDHE+RSA+AES128+SHA256'),\
          ('ECDHE+ECDSA+3DES+SHA'),('ECDHE+RSA+AES256+SHA384'),\
          ('ECDHE+RSA+AES256+SHA'),\
          ('ECDHE+RSA+AES256'),('ECDHE+ECDSA+AES128+SHA256'),\
          ('RSA+AES128+SHA256'),\
          ('ECDHE+ECDSA+AES128+SHA'),\
          ('RSA+AES128+SHA'),('RSA+AES128+SHA256') ]

ciphers=[('ECDH+AESGCM'),('ECDH+ECDSA'),('RSA+AES+CCM'),\
         ('ECDH+AESCBC'),('RSA'),('ECDH+3DESCBC'),\
         ('ECDH+ECDSA+AESCBC'),('RSA+AESCCM'),\
         ('DHE+DESCBC'),('ECDH+AESCBC'),\
         ('RSA+AES256+SHA'),('DHE+AES256+SHA384'),\
         ('RSA+AES128+SHA256'),('ECDHE+ECDSA+3DES+SHA'),\
         ('ECDHE+AES256+SHA384'),('RSA+CAMELLIA256+SHA'),\
         ('DHE+AES256+SHA'),('ECDHE+AES256+SHA'),\
         ('ECDHE+ECDSA+AES128+SHA256'),('DHE+CAMELLIA128+SHA'),\
         ('RSA+AES128+SHA256'),('DHE+CAMELLIA128_SHA256'),\
         ('RSA+CAMELLIA256+SHA256'),('ECDHE+ECDSA+AES128+SHA'),\
         ('RSA+AES128+SHA'),('ECDH+AES256+SHA384')]
'''

ciphers = [('ECDH+AESGCM'),('DH+AESGCM'),('ECDH+AES256'),('DH+AES256'),('ECDH+AES128'),('DH+AES'),('ECDH+HIGH'),\
    ('DH+HIGH'),('ECDH+3DES'),('DH+3DES'),('RSA+AESGCM'),('RSA+AES'),('RSA+HIGH'),('RSA+3DES'),\
    ('ECDHE+ECDSA+AES256'),('ECDHE+ECDSA+3DES'),('ECDHE+ECDSA+AES128'),\
    ('DH+CAMELLIA'),('RSA+AESGCM')]



class DESAdapter(HTTPAdapter):
    """
    A TransportAdapter that re-enables 3DES support in Requests.
    """
    def __init__(self,cipher_suite):
        self.cipher_suite=cipher_suite
        super(DESAdapter,self).__init__()

    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=self.cipher_suite)
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=CIPHERS)
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).proxy_manager_for(*args, **kwargs)

both_reachable=pd.read_csv('./crawl_both_reachable.csv',header=None)
both_websites=both_reachable.iloc[:,0]
TLSv2_websites=pd.read_csv('./TLSv1.2_websites2.csv',\
                                 header=None)

def validate_ciphersuite():
    global ciphers
    for cipher_suite in ciphers:
        try:
          #if the cipher is valid.
          ssl_ctxt= DESAdapter(cipher_suite)
          s = requests.Session()
          s.mount('https://www.google.com', ssl_ctxt)
          r = s.get('https://www.google.com')
        except:
          import sys
          if(sys.exc_info()[1][0]=='No cipher can be selected.'):
             print("Cipher suite:",cipher_suite)
             print("Invalid cipher")

validate_ciphersuite()
cipher_suite_times=defaultdict(list) #this is of the form <company_name>:[list of ciphersuite times]

#outer loop for cipher suite.
#inner loop for websites.
#All the suites are supported/
for website in TLSv2_websites.values:
    website=website[0]
    #print("website: ",website)
    http_times=[]
    for time_ in range(10):
        try:
          start=time.time()
          r = requests.get('http://www.'+website)
          end=time.time()
          http_times.append(end-start)
        except:
          import sys
          print(sys.exc_info()[1])
          #when the suite is not supported.
          http_times.append(100)   

    cipher_suite_times[(website,'http')].append(sum(http_times)/len(http_times))
    print((website,'http'),sum(http_times)/len(http_times))

    with open('crawl_cipher_suites.csv','a') as fp:
         fp.write("%s,%s,%0.4f\n"%(website,'http',\
                       sum(http_times)/len(http_times)))     
         fp.close()
    
    for cipher_suite in ciphers:
        #print("Cipher suite:",cipher_suite)
        access_times=[]
        ssl_ctxt=DESAdapter(cipher_suite)       
        for time_ in range(10):
            start=time.time()
            s = requests.Session()
            try:
              s.mount('https://www.'+website, ssl_ctxt)
              r = s.get('https://www.'+website)
              end=time.time()
              access_times.append(end-start)
            except:
              #when the suite is not supported.
              #import sys
              #print(sys.exc_info()[1])
              access_times.append(100)   

        cipher_suite_times[(website,cipher_suite)].append(sum(access_times)/len(access_times))
        print((website,cipher_suite),sum(access_times)/len(access_times))
        with open('crawl_cipher_suites.csv','a') as fp:
             fp.write("%s,%s,%0.4f\n"%(website,cipher_suite,\
                       sum(access_times)/len(access_times)))     
             fp.close()    
               


