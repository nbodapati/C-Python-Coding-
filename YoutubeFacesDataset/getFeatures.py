import os
import sys
import numpy as np
import pandas as pd
from scipy import misc
import skimage.transform as trf
import pickle
main_path='./aligned_images_DB/'

#append with each image
#[batch_size,h,w,c]
Images=np.zeros((1,64,64,3))
#names of folders
Labels=[0]

def get_list_folders(main_folder):
    return sorted(os.listdir(main_folder))

def load_files_folder(main_folder,folder):
    #append with each image
    #[batch_size,h,w,c]
    Images=np.zeros((1,64,64,3))
    #names of folders
    Labels=['None']
    subfolders=os.listdir(main_folder+folder)
    #video frame directories
    for sf in subfolders:
       files_list=os.listdir(main_folder+folder+sf+'/')
       for file_ in files_list:
           print(folder+sf+'/'+file_)
           #img=Image.open(folder+sf+'/'+file_)
           #print(img.size,img.format)
           img=misc.imread(main_folder+folder+sf+'/'+file_)
           print(img.shape,img.dtype)
           #resize all to (224,224,3) - either upscale/downascle
           #enable anti-aliasing so that no artifacts during downscaling.
           img=trf.resize(img,(64,64,3), mode='reflect')
           #print(img.shape,img.dtype)
           img=np.expand_dims(img,axis=0)
           if(Images.shape[0]>100):
             return Images[1:,:,:,:],Labels[1:]

           Images=np.append(Images,img,axis=0)
           print(img.shape,img.dtype,Images.shape,folder[:-1])
           Labels.append(folder[:-1]) #ignore / at the end.  

    #the first entry is not useful.
    return Images[1:,:,:,:],Labels[1:]
 
if __name__ =='__main__':
   folders=get_list_folders(main_path)
   #list of names
   #get first 100 frames from top-100
   i=0
   for folder in folders[:150]:
       Images,labels=load_files_folder(main_path,folder+'/')     
       #After processing each celebrity images 
       #store the results to .mat file which says 
       #YTF_dataset.mat
       #mat_dict={'Images':Images.reshape(-1,64*64*3).tolist(),'Labels':Labels}
       #io.savemat(file_name='YTF_dataset.mat',mdict=mat_dict,appendmat=False)
       #pickle.dump(mat_dict,open('YTF_dataset.pkl','wb'))
       #df=pd.DataFrame.from_dict(mat_dict)
       df=pd.DataFrame(Images.reshape(-1,64*64*3))
       df['Labels']=labels 
       df.to_csv('./YTF_dataset.csv',mode='a') 
       i+=1
       print("Done with saving..",i)
         
