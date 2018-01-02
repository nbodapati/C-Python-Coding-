from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import sys

#run a loop from 0 to 10000 
#create a dict with mapping from file number to title
#store the files in Gutenberg_Corpus/ with filenumber.txt
#pickle the dictionary as Guteberg_meta.pkl

for file_num in range(11,10000):
    try:
       text = strip_headers(load_etext(file_num)).strip()
       title=text.split('\n')[0]
       print(file_num,title)
    except:
       print("Error:",sys.exc_info()[1])

