
# Author: Abhijnan Nath
# Run an end-to-end event coreference resolution experiment using the GVC annotations, only works for lemman currently.
import os
import pickle
import sys
sys.path.insert(0, os.getcwd())

from parsing.parse_gvc import extract_mentions
#from bert_stuff import *
from bert_stuff_crossencoder import *
# from bert_stuff import generate_cdlm_embeddings
import argparse

from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from evaluations.eval import *
import numpy as np
from incremental_clustering import incremental_clustering
from sklearn.cluster import AgglomerativeClustering


def get_cosine_similarities(mention_pairs, vector_map):
    """
    Calculate the cosine similarity as row wise dot product of 2 arrays of vectors

    Parameters
    ----------
    mention_pairs: list
    vector_map: dict

    Returns
    -------
    list
    """
    def normed(a):
        return a/np.linalg.norm(a, axis=1).reshape((-1, 1))
    m1s, m2s = zip(*mention_pairs)
    lhs = np.array([vector_map[m].detach().cpu().numpy() for m in m1s])
    rhs = np.array([vector_map[m].detach().cpu().numpy() for m in m2s])

    return np.sum(normed(lhs) * normed(rhs), axis=1)


def get_mention_pair_similarity_cdlm_bi(mention_pairs, mention_map, relations, working_folder):
    """
    Generate similarity using CDLM as a bi-encoder, i.e., generate mention embeddings and calculate
    cosine similarity between the mention_pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: list
    relations: list
    working_folder: str

    Returns
    -------
    list
    """
    # get the CDLM embeddings for the mentions
    vec_map_path = working_folder + '/cdlm_vec_map.pkl'
    # # if the vector map pickle file does not exist, generate the embeddings
    if not os.path.exists(vec_map_path):
        # use the appropriate key name i.e. bert_doc for longer documents and bert_sentence for sentences
        #use either cdlm bi encoder embeddings or cross encoder embeddings 
        cdlm_vec_map= generate_cdlm_embeddings(mention_map, vec_map_path, key_name='bert_doc', num_gpus=4, batch_size=20, cpu=False)
        #generate_cross_cdlm_embeddings(mention_pairs, mention_map, vec_map_path, key_name='bert_doc', num_gpus=4, batch_size=150, cpu=False)
    # # read the vector_map pickle
    #cdlm_vec_map = pickle.load(open(vec_map_path, 'rb'))
    # # generate and return the cosine similarities
    #return get_cosine_similarities(mention_pairs, cdlm_vec_map)
    return cdlm_vec_map 
def get_mention_pair_similarity_cdlm_cross(mention_pairs, mention_map, relations, working_folder):
    """
    Generate similarity using CDLM as a cross-encoder, i.e., generate mention embeddings and calculate
    pairwise scoring between embeddings of mention_pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: list
    relations: list
    working_folder: str

    Returns
    -------
    list
    """
    # get the CDLM embeddings for the mentions
    vec_map_path = working_folder + '/cdlm_vec_map_crossencoder.pkl'
    # # if the vector map pickle file does not exist, generate the cross embeddings and get pairwise cdlm scores
    if not os.path.exists(vec_map_path):
        # use the appropriate key name i.e. bert_doc for longer documents and bert_sentence for sentences
         
        generate_cross_cdlm_embeddings(mention_pairs, mention_map, vec_map_path, key_name='bert_doc', num_gpus=4, batch_size=100, cpu=False)
    # # read the vector_map pickle
    cdlm_scores = pickle.load(open(vec_map_path, 'rb'))
    
    return cdlm_scores

 def get_mention_pair_similarity_lemma_GVC(mention_pairs, mention_map,working_folder):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
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
def cluster_agglo(affinity_matrix, threshold=0.5):
    """
    Agglomerative clustering based on the affinity matrix
    :param affinity_matrix: np.array
        The similarity matrix. Need to convert into distance matrix for agglo clustering algo
    :param threshold: float
        Linkage threshold
    :return: list, np.array
        The labels of the nodes
    """
    clustering_ = AgglomerativeClustering(n_clusters=None,
                                          affinity='precomputed',
                                          linkage='average',
                                          distance_threshold=threshold)
    # convert affinity into distance
    distance_matrix = 1 - np.array(affinity_matrix)
    # fit predict
    labels = clustering_.fit_predict(distance_matrix)
    # get clusters
    clusters = defaultdict(list)
    for ind, label in enumerate(labels):
        clusters[label].append(ind)
    return list(clusters.values()), labels


def cluster_cc(affinity_matrix, threshold=0.3):
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

 
def coreference(curr_mention_map, all_mention_map, working_folder,
                men_type='evt', sim_type='lemma',
                cluster_algo='cc', threshold=0.9):
    """

    Parameters
    ----------
    curr_mention_map
    all_mention_map
    working_folder
    men_type
    relations
    sim_type
    cluster_algo: str
        clustering algorithm, options: ['cc', 'agglo', 'inc']
    threshold: double
    Returns
    -------

    """

    # sort event mentions and make men to ind map
    curr_mentions = sorted(list(curr_mention_map.keys()), key=lambda x: curr_mention_map[x]['m_id'])
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
            for j in range(i + 1):
                if i != j:
                    mention_pairs.append((list_mentions[i], list_mentions[j]))

    # get the similarities of the mention-pairs from either lemmas or cdlm embeddings or pairwise cdlm scores for current mention map 
    if sim_type == 'lemma':
        #similarities = get_mention_pair_similarity_lemma(mention_pairs, all_mention_map, relations, working_folder)
        similarities = get_mention_pair_similarity_lemma_GVC(mention_pairs, all_mention_map, working_folder) #for GVC
        
    elif sim_type == 'cdlm':
        similarities = get_mention_pair_similarity_cdlm_bi(mention_pairs, all_mention_map, relations, working_folder)
        
    elif sim_type == 'cross-encoder': 
        cdlm_scores = get_mention_pair_similarity_cdlm_cross(mention_pairs, curr_mention_map, relations, working_folder)
        #cdlm_vec_map, tokens = get_mention_pair_similarity_cdlm_cross(mention_pairs, curr_mention_map, relations, working_folder)
        #similarities = [x for x in similarities.values()] #returns a dict with pairwise mention_ids as key, so only use scores 
        #print(len(similarities ))

    # get indices
    mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(curr_mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    # clustering algorithm and mention cluster map
    if cluster_algo == 'cc':
        clusters, labels = cluster_cc(similarity_matrix, threshold)
    elif cluster_algo == 'agglo':
        clusters, labels = cluster_agglo(similarity_matrix)
    elif cluster_algo == 'inc':
        clusters, labels = incremental_clustering(similarity_matrix, threshold, curr_mentions,
                                                  all_mention_map, curr_men_to_ind)
    else:
        raise AssertionError
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)
    return similarities
     


def run_coreference(ann_dir, source_dir, working_folder, men_type='evt'):
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

 
    # extract the mention maps for GVC corpus 
    eve_mention_map, doc_sent_map = extract_mentions(ann_dir, source_dir, working_folder)
    print(len(eve_mention_map), len(doc_sent_map))
    
    # which coreference mention map, only 'evt' for GVC since it doesn't contain entities. 
    if men_type == 'evt':
        curr_mention_map = eve_mention_map
    else:
        curr_mention_map = ent_mention_map

    # create a single dict for all mentions, don't include entities for GVC since there are none!
 
    all_mention_map = {**eve_mention_map} #only events in case of GVC
    
    
    #do some filtering:
   
    curr_mention_map_new = {}
    for key, mention in curr_mention_map.items():
        mention_text = mention['mention_text']
        if len(mention_text.strip()) > 2 and len(mention_text.split()) < 4:
            curr_mention_map_new[key] = mention
   
    #run pipeline without filtering 
 
    #coreference(curr_mention_map, all_mention_map, working_folder, men_type,
#                 sim_type='lemma', cluster_algo='cc')
    #run pipeline with filtering 

    coreference(curr_mention_map_new, all_mention_map, working_folder, men_type, 
                sim_type='lemma', cluster_algo='cc')
    
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run and evaluate cross-document coreference resolution on GVC Annotations')
    parser.add_argument('--ann', '-a', help='Path to the GVC Gold Annotation Directory')
    parser.add_argument('--source', '-s', help='Path to the GVC Verbose meta-data Directory')
    parser.add_argument('--tmp_folder', '-t', default='./tmp', help='Path to a working directory')
    parser.add_argument('--men_type', '-m', default='evt', help='Mention type for coreference. Only evt for GVC corpus')
    args = parser.parse_args()
    print("Using the Annotation Directory: ", args.ann)
    print("Using the Source     Directory: ", args.source)
    print("Using the Working    Directory:", args.tmp_folder)
    run_coreference(args.ann, args.source, args.tmp_folder)
   
