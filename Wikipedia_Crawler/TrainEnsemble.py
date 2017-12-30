from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict

df=pd.read_csv('./Optimal_Phrases.csv')
phrases=df['optimal_phrases']
labels=df['quality']

vocab=defaultdict()

def build_vocab():
