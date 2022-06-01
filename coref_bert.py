#!/usr/bin/env python
# coding: utf-8

# In[17]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Rehan
# Run an end-to-end event coreference resolution experiment using the LDC annotations
# import sys
# sys.path.append('.')


#from parse_ldc import extract_mentions
from parsing.parse_ldc import *
from cdlm_bert_models.cross_encoder import FullCrossEncoder, FullCrossEncoderSingle

import argparse


import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *
from bert_stuff import *
import numpy as np
import sys
#sys.argv = ['']
sys.path.append('.')
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from evaluations.eval import *
import os
import copy
import configparser
config = configparser.ConfigParser()
config.read('model_parameters.ini')


def get_mention_pair_similarity_lemma(mention_pairs, mention_map, relations, working_folder):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    similarities = []

    # generate similarity using the mention text
    for pair in mention_pairs:
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['mention_text']
        men_text2 = men_map2['mention_text']
        similarities.append(int(men_text1 == men_text2))

    return similarities
def get_mention_pair_cosinesimilarity(mention_pairs, mention_map,vec_map, relations, working_folder):
    """
    Generate the cosine similarities for mention pairs using the CDLM model 

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    evt_vec_map : dict with mention IDs as keys and CDLM Embeddings as values
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    
    
    
    
    similarities = []
    sims_list = []
    cos_s = torch.nn.CosineSimilarity()
      # generate similarity using the mention text
    for i, pair in enumerate(mention_pairs):
         
        men1, men2 = pair
        men_vec1 = torch.tensor(vec_map[men1].reshape(1,-1)) 
        men_vec2 = torch.tensor(vec_map[men2].reshape(1,-1))
        sims = cos_s(men_vec1,men_vec2).data.cpu().numpy()
        
        
        #sims[sims>0.98757007] = 1  #mean for cdlm model 
        #sims[sims<=0.98757007] =0
#         sims[sims>0.95] = 1
#         sims[sims<=0.95] =0
        #sims[sims>0.47] = 1  # mean for coref bert model 
        #sims[sims<=0.47] =0
#         sims[sims>0.39636973] = 1  # mean for bert large  model 
#         sims[sims<=0.39636973] =0
        
#         sims[sims>0.6] = 1  #  
#         sims[sims<=0.6] =0
        
        #sims[sims>0.7] = 1  #  
        #sims[sims<=0.7] =0
        #sims_list.append([pair, sims])
        similarities.append(sims)
    similarities = np.asarray(similarities)
    sim_mean = np.mean(similarities)
    similarities[similarities >sim_mean] = 1
    similarities[similarities <=sim_mean] = 0
    print(sim_mean)   

    return similarities 




def cluster_cc(affinity_matrix, threshold=0.8):
    """
    Find connected components using the affinity matrix and threshold -> adjacency matrix
    Parameters
    ----------
    affinity_matrix: np.array
    threshold: float

    Returns
    -------
    list, np.array
    """
    adjacency_matrix = csr_matrix(affinity_matrix > threshold)
    clusters, labels = connected_components(adjacency_matrix, return_labels=True, directed=False)
    return clusters, labels


def run_coreference(ann_dir, source_dir, working_folder,bert_model, men_type='evt', bert=False):
    """
    Run the coreference resolution pipeline on LDC annotations in the following steps:
        1) Read and parse the annotations + source documents
        2) Generate pairs of mentions from the same topic
        3) Generate similarities for the pairs and create a similarity matrix
        4) Agglomeratively cluster using the similarity matrix
        5) Evaluate the clusters against the gold standard clusters (BCUB, MUC, CEAF, CONNL)

    Parameters
    ----------
    ann_dir: str
    source_dir: str
    working_folder: str
    men_type: str

    Returns
    -------
    None
    """
    # create working folder
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
     

    # extract the mention maps
    eve_mention_map, ent_mention_map, relations, doc_sent_map = extract_mentions(ann_dir, source_dir, working_folder)
    eve_vector_map ={}
    
    #change the event or entity mention map to include large documents for bert models as well as cdlm 

    # which coreference mention map and getting longer documents for mention type
    curr_mention_map = eve_mention_map
    if bert==False:
        if men_type == 'evt':
            curr_mention_map = eve_mention_map
        elif men_type == 'ent':
            
            curr_mention_map = ent_mention_map
            print(len(curr_mention_map))
        else:
            curr_mention_map = ent_mention_map
            print(len(curr_mention_map))
            
                





 

    print(bert, men_type)
    if bert==True:
        if men_type == 'evt':
            print('event---true')
            
       

            eve_mention_map = create_longdocs(doc_sent_map , eve_mention_map)
            curr_mention_map = eve_mention_map
            #c_mention_map = eve_mention_map

            men_ids, bert_sentences = zip(*[(men_id, men['bert_doc']) for men_id, men in eve_mention_map.items()])
            #bert_sentences = bert_sentences[0:30]
            x = generate_cdlm_embeddings(config,working_folder,bert_model, bert_sentences,men_type = men_type, num_gpus=4, batch_size=50, cpu=False)

            eve_vector_map = {m_id: vec for m_id, vec in zip(men_ids, x)}
            print('created event vector maps')
    
        else:
            print('entity---true')

            ent_mention_map = create_longdocs(doc_sent_map , ent_mention_map)
            curr_mention_map = ent_mention_map
            #c_mention_map = ent_mention_map
            men_ids, bert_sentences = zip(*[(men_id, men['bert_doc']) for men_id, men in ent_mention_map.items()])
            #bert_sentences = bert_sentences[0:30]
            x = generate_cdlm_embeddings(config,working_folder,bert_model, bert_sentences,men_type = men_type, num_gpus=4, batch_size=50, cpu=False)
            ent_vector_map = {m_id: vec for m_id, vec in zip(men_ids, x)}
            #print(len(ent_vector_map))
            print('created entity vector maps')
    
   
                          
    #print(eve_mention_map['VMIC0015RNR.000397']['bert_doc'])
    #print(eve_mention_map.keys())
    #print(len(eve_vector_map))

    # create a single dict for all mentions
    all_mention_map = {**eve_mention_map, **ent_mention_map}

    # sort event mentions and make men to ind map
    curr_mentions = sorted(list(curr_mention_map.keys()), key=lambda x: curr_mention_map[x]['m_id'])
    #curr_mentions = sorted(list(c_mention_map.keys()), key=lambda x: c_mention_map[x]['m_id'])
    curr_men_to_ind = {eve: i for i, eve in enumerate(curr_mentions)}

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, all_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = working_folder + f'/{men_type}_gold.keyfile'
    generate_key_file(curr_gold_cluster_map, men_type, working_folder, gold_key_file)

    # group mentions by topic
    topic_mention_dict = defaultdict(list)
    for men_id, coref_map in curr_mention_map.items():
        topic = coref_map['topic']
        topic_mention_dict[topic].append(men_id)

    # generate mention-pairs
    mention_pairs = []
    for mentions in topic_mention_dict.values():
        list_mentions = list(mentions)
        for i in range(len(list_mentions)):
            for j in range(i+1):
                if i != j:
                    mention_pairs.append((list_mentions[i], list_mentions[j]))
    print(len(mention_pairs))

    # get the similarities of the mention-pairs
    similarities = get_mention_pair_similarity_lemma(mention_pairs, all_mention_map, relations, working_folder)
    
    # get the cosine similarities of the mention-pairs from their respective embeddings
    
    if bert==True:
        
        if men_type =='evt':
            print("getting cosine similarities for events ")
            similarities = get_mention_pair_cosinesimilarity(mention_pairs, curr_mention_map,eve_vector_map, relations, working_folder)
            similarities = [int(x) for x in similarities ]
    
        else:
            print("getting cosine similarities for entities ")
            similarities = get_mention_pair_cosinesimilarity(mention_pairs, curr_mention_map,ent_vector_map, relations, working_folder)
            similarities = [int(x) for x in similarities ]
        
    #x = generate_cdlm_embeddings(config,working_folder,bert_model, bert_sentences,men_type = men_type, num_gpus=4, batch_size=15, cpu=False)
    
    
    
    
    

    # get indices
    mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in mention_pairs]
    row, col = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(curr_mentions)
    similarity_matrix = csr_matrix((similarities, (row, col)), shape=(n, n)).toarray()

    # clustering algorithm and mention cluster map
    clusters, labels = connected_components(similarity_matrix)
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("coref_bert_new.py",description='Run and evaluate cross-document coreference resolution on                                                  LDC Annotations')
    parser.add_argument('--ann', '-a', default='/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/LDCann', help='Path to the LDC Annotation Directory')
    parser.add_argument('--source', '-s',default='/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/LDCsource', help='Path to the LDC Source Directory')
    parser.add_argument('--tmp_folder', '-t', default='/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/tmp_folder', help='Path to a working directory')
    
    parser.add_argument('--model', default='bert_base', help='Transformer model type used for coref')
    parser.add_argument('--men_type', '-m', default='ent', help='Mention type for coreference. Either evt or ent')
    parser.add_argument('--bert', '-bert', default=False,action='store_true', help='Whether to use transformer embedding or lemma for coref')
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print("Using the Annotation Directory: ", args.ann)
    print("Using the Source     Directory: ", args.source)
    print("Using the Working    Directory:", args.tmp_folder)
    
    print("Using the Transformer model :", args.model)
    print("Mention type for coreference. Either evt or ent:", args.men_type)
    
    print("Whether using embedding or lemma :", args.bert)
    #run_coreference('/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/LDCann', '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/LDCsource', '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/tmp_folder', 'bert_large','evt', bert=True )
    run_coreference(args.ann, args.source, args.tmp_folder, args.model,args.men_type , args.bert )


# In[ ]:





# In[ ]:




