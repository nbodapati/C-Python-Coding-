import pandas as pd
import numpy as np
from scipy import misc
import skimage.transform as trf
import os
import sys
import pickle

def weighted_RGB2Grey(img):
    #(100,100,3) - image[0][0] - gives the [R,G,B] values
    new_img=np.zeros((img.shape[0],img.shape[1]))
    for rownum in range(img.shape[0]):
        for colnum in range(img.shape[1]):
            rgb=img[rownum][colnum]    
            new_img[rownum][colnum]=0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
    #new_img=new_img.flatten().reshape()
    return new_img #[100,100] shape        

def get_patches(new_img):
    #new_img - shape [100,100]
    #collect patches of size 20x20 at 10pixel space
    #Number of patches: 100
    patches=np.zeros((1,400))
    for rownum in range(0,new_img.shape[0]-20+1,10):
        for colnum in range(0,new_img.shape[1]-20+1,10):
            patch=new_img[rownum:rownum+20,colnum:colnum+20]
            patch=patch.flatten().reshape(1,-1) 
            patches=np.append(patches,patch,axis=0)
    print("Patches shape: ",patches.shape)
    return patches[1:,:]
        

patches=np.zeros((1,400))
files_list=os.listdir('./Images/')
 
image_number=0
for file_ in files_list:
    image_number+=1
    print('./Images/'+file_,image_number)
    try:
       img=misc.imread('./Images/'+file_)
    except:
       import sys
       print(sys.exc_info()[1])
       continue
    img=trf.resize(img,(100,100,3), mode='reflect')
    grey_img=weighted_RGB2Grey(img)
    patches=np.append(patches,get_patches(grey_img),axis=0)
    print("Total shape..",patches.shape)
patches=patches[1:,:]
pickle.dump(patches,open('patches.pkl','wb'))   
