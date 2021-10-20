#!/usr/bin/env python
# coding: utf-8

# In[55]:


#read in all embeddings
import numpy as np
import glob
import timeit
import pandas as pd
import numpy as np
import hnswlib
import torch
import os
import psutil
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize


# In[58]:


#cd /Users/liushaoyu/Documents/embeddings/


# In[29]:


"""numpy_vars = []
for np_name in glob.glob('/Users/liushaoyu/Documents/embeddings/*.npy'):
    numpy_vars.append(np.load(np_name))
dataset = np.vstack(numpy_vars)
"""


# In[59]:


dataset = np.load('data_algo.npy')


# In[60]:


print('Normalizing the dataset')
dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
print('Done')


# In[61]:


assert dataset.dtype == np.float32


# In[63]:


# Choose random data points 1% to be queries.
number_of_queries = 1000
print('Generating queries')
np.random.seed(4057218)
np.random.shuffle(dataset)
queries = dataset[len(dataset) - number_of_queries:]
dataset = dataset[:len(dataset) - number_of_queries]
print('Done')


# In[64]:


queries.shape, dataset.shape


# In[65]:


print('Solving NN queries using linear scan')
t1 = timeit.default_timer()
answers = []
for query in queries:
    answers.append(np.dot(dataset, query).argmax())
t2 = timeit.default_timer()
print('Done')
print('Linear scan average time: {} per query'.format((t2 - t1) / float(
    len(queries))))
print('Linear scan total time: {}'.format((t2 - t1)))
linear_scan_time = t2 - t1


# In[66]:


def HNSW(efc, ef, dim, num_elements, M, k = 1, distance = 'cosine'):   
    """
    :param ef_construction: controls the index_time/index_accuracy tradeoff. 
    Bigger ef_construction leads to longer construction, but better index quality
    :param dim: size of embeddings
    :param num_elements: number of elements to iterate for looking for nearest neighbors
    :param M: the number of bi-directional links created for every new element during construction
    :param k: number of nearest neighbors k >= 1
    :param distance: possible options are 'l2', 'cosine' or 'ip'
    """
    p = hnswlib.Index(space= distance, dim=dim)  # possible options are l2, cosine or ip
    """Initing index
       max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded during insertion of an element.
    The capacity can be increased by saving/loading the index, see below.
    #
       ef_construction - controls index search speed/build speed tradeoff
    #
       M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
        Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    """
    p.init_index(max_elements=50000, ef_construction=efc, M = M)

    p.set_ef(ef)

    p.set_num_threads(4)

    print("Adding %d dataset elements" % (len(dataset)))
    
    p.add_items(dataset)

    # Query the elements for themselves and measure recall:
    t1 = timeit.default_timer()
    labels, distances = p.knn_query(queries, k=1)
    t2 = timeit.default_timer()
    
    
    print('HNSW time: {} per query'.format((t2 - t1) / float(
        len(queries))))
    print("Recall for HNSW batch:", np.mean(labels.reshape(-1) == np.array(answers)), "\n")
    
    return t2-t1, np.mean(labels.reshape(-1) == np.array(answers))


# **Relationship between ef and recall**

# In[67]:


dim = dataset.shape[1]
num_elements = 1000

ef_trial = [1,10,15,20,25,30,40,50,60,70,80,90,100,200]
query_time_list = []
recall_list = []
construct_time_list = []
mem_use_list = []
for item in ef_trial:
    
    m1 = py.memory_info()[0]/2.**30
    t1 = timeit.default_timer()
    query_time, recall = HNSW(efc = 100, ef = item, dim = dataset.shape[1], num_elements = 1000, M = 48, k = 1, distance = 'cosine')
    t2 = timeit.default_timer()
    m2 = py.memory_info()[0]/2.**30
    
    construct_time = t2-t1
    mem_use = m2 - m1
    query_time_list.append(query_time)
    recall_list.append(recall)
    construct_time_list.append(construct_time)
    mem_use_list.append(mem_use)
    
df = pd.DataFrame({'query_time': query_time_list, 'recall': recall_list, 'construct_time': construct_time_list, 'memory_use': mem_use_list})
df['linear_scan_time'] = linear_scan_time


# In[68]:


df.to_csv('/Users/liushaoyu/Documents/embeddings/hnsw_efc_100.csv')

