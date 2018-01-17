import pandas as pd
import numpy as np
import pickle
import requests
import random
import sys
from collections import defaultdict

neighbors=pickle.load(open('neighbors_plant.pkl','rb'))

glosses=[]
wnids=[]
words=[]
cdists=[]

for d_ in neighbors:
    glosses.append(d_['gloss'])
    wnids.append(d_['wnid'])
    words.append(d_['words'])
    cdists.append(d_['cdist'])
    
#get image files associated with each wnid.
def get_images(wnid):
    image_names_urls='http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=%s'%wnid
    names_urls= requests.get(image_names_urls).content.decode('utf-8')
    names_urls=names_urls.split('\r\n')
    def map_names_urls(x):
        x=x.split()
        #print(x)
        return x[1]#{'name':x[0],'url':x[1]}
    if(len(names_urls)):
       print(names_urls)
 
    names_urls=[map_names_urls(x) for x in names_urls if x!='']
    return names_urls

wnid_images=defaultdict(list)
for wnid in wnids:
    #get images
    wnid_images[wnid]=get_images(wnid)

pickle.dump(wnid_images,open('wnid_images.pkl','wb'))
    
