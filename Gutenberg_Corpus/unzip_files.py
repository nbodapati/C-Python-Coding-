import zipfile
import os

path_to_files='./text/'
dest_path='./text/'
list_of_files=os.listdir(path_to_files)
list_of_files=sorted(list_of_files)

for file_ in list_of_files:
    with zipfile.ZipFile(os.path.join(path_to_files,file_),"r") as zip_ref:
         zip_ref.extractall(dest_path)
