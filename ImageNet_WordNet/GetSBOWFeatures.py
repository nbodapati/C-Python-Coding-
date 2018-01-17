#First download all the images associated with the urls.
#walk through the synsets using their wnids and get lists of 
#urls - load into a dictionary.

import requests
from  urllib.request import urlretrieve
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import pickle

download_folder='./Images/'
'''
x=pd.read_csv('./words_wnids_gloss_imagenet.csv')
print(x.shape)
xml_dataframe=x.dropna(how='any')
xml_dataframe=xml_dataframe.drop(labels=['Unnamed: 0'],axis=1)
print(xml_dataframe.shape)
print(xml_dataframe.head())
'''
#For now only get the neighbors.
neighbors=pickle.load(open('neighbors_plant.pkl','rb'))

wnid_list=[]
for d_ in neighbors:
    wnid_list.append(d_['wnid'])

print(wnid_list)
wnids_urls_dict=defaultdict(list)
#all those that have finite number of images.
all_wnids=[]
#all urls
all_urls=[]
#all paths to jpgs
all_images=[]

wnids_no_urls=[]
#map urls to path to jpgs
downloaded_images_map=defaultdict(str)
downloaded_urls_map=defaultdict(str)

#some wnids have associated urls while some dont.
#Only retain those that have.
#make a list of those that dont.
#get the url.
def map_names_urls(x):
    if(len(x)==0):
      return '' #empty url.
    x=x.split()
    return x[1]#{'name':x[0],'url':x[1]}

image_number=0
for wnid in wnid_list:
    image_urls='http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=%s'%wnid
    print("Image url..",image_urls)
    names_urls= requests.get(image_urls).content.decode('utf-8')
    names_urls=names_urls.split('\r\n')
    urls=list(map(map_names_urls,names_urls))
   
    wnids_urls_dict[wnid]=urls

    if(len(urls)):
       all_wnids.append(wnid)
    else:
       wnids_no_urls.append(wnid)
        
    for url in urls: 
        if(url==''):
           continue
        try:
           urlretrieve(url, os.path.join(download_folder+str(image_number)+".jpg"))
           all_urls.append(url) 
           all_images.append(os.path.join(download_folder+str(image_number)+".jpg"))
           image_number+=1
           print("Url..",url)
           print("Image..",os.path.join(download_folder+str(image_number)+".jpg"))  
           downloaded_images_map[url]=os.path.join(download_folder+str(image_number)+".jpg")
           downloaded_urls_map[os.path.join(download_folder+str(image_number)+".jpg")]=url
           print("Number of images downloaded.. ",image_number)
        except:
           import sys
           print(sys.exc_info()[1]) 

mdict=dict(wnids_urls_dict=wnids_urls_dict,all_wnids=all_wnids,all_urls=all_urls,all_images=all_images,\
           wnids_no_urls=wnids_no_urls,\
           downloaded_images_map=downloaded_images_map,downloaded_urls_map=downloaded_urls_map)
pickle.dump(mdict,open('loaded_images.pkl','wb'))


  
