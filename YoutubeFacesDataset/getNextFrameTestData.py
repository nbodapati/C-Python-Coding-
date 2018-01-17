import os
import sys
import numpy as np
import pandas as pd
from scipy import misc
import skimage.transform as trf
import pickle
main_path='./aligned_images_DB/'

h=64
w=64
c=3

def get_list_folders(main_folder):
    return sorted(os.listdir(main_folder))

def load_files_folder(main_folder,folder):
    #append with each image
    #[batch_size,h*w*c]
    Images=np.zeros((1,h*w*c))
    #names of folders
    Labels=['None']
    subfolders=os.listdir(main_folder+folder)
    #video frame directories
    for sf in subfolders:
       files_list=os.listdir(main_folder+folder+sf+'/')
       #to have a high diversity - only use frames/videos that are no longer than 
       #100 frames
       #This way each valid celeb gets atmost 20datapoints per video sequence. 
       if(len(files_list)//10>0 and len(files_list)>110):
         continue
       print("*****************************************Valid folder: ",folder+sf)
    
       for file_ in files_list:
           print(folder+sf+'/'+file_)
           #img=Image.open(folder+sf+'/'+file_)
           #print(img.size,img.format)
           img=misc.imread(main_folder+folder+sf+'/'+file_)
           print(img.shape,img.dtype)
           #resize all to (64,64,3) - either upscale/downascle
           #enable anti-aliasing so that no artifacts during downscaling.
           img=trf.resize(img,(h,w,c), mode='reflect')
           img=img.reshape(-1,h*w*c) 
           #use the first 4 frames for time steps and 5th as prediction 
           #only use the first-10*n frames.
           Images=np.append(Images,img,axis=0)
           print(img.shape,img.dtype,Images.shape,folder[:-1])
           Labels.append(folder[:-1]) #ignore / at the end.  
           if(Images.shape[0]==(len(files_list)//10)*10+1):
             return (Images[1:,:],Labels[1:])

    #the first entry is not useful.
    #if no valid folder - this returns empty. 
    return Images[1:,],Labels[1:]
 
if __name__ =='__main__':
   folders=get_list_folders(main_path)
   #list of names
   #get first 100 frames from top-100
   i=0
   len_=0
   for folder in folders[300:400]:
       Images,labels=load_files_folder(main_path,folder+'/')     
       if(len(labels)==0):
         continue

       len_+=len(labels) 
       print(len_)
       df=pd.DataFrame(Images)
       if(labels[0]=='Labels'):
          assert()
       df['Labels']=labels 
       df.to_csv('./YTF_dataset_nextf_test.csv',mode='a') 
       i+=1
       print("Done with saving..",i)
         


