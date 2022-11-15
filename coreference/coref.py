# Author: Rehan
# Run an end-to-end event coreference resolution experiment using the LDC annotations
import os
import pickle
import sys
sys.path.insert(0, os.getcwd())
from parsing.parse_ldc import extract_mentions
from bert_stuff_crossencoder import * # for importing cdlm cross-encoder
# from bert_stuff import generate_cdlm_embeddings  # for importing cdlm-bi-encoder
import argparse

from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from evaluations.eval import *
import numpy as np
from coreference.incremental_clustering import incremental_clustering
from sklearn.cluster import AgglomerativeClustering


def load_data_pair_coref(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            #triviality_label = int(row[3])
            triviality_label = 0 if row[3] =='NEG' else 1
            
            all_examples.append((mention_pair, triviality_label))

    return all_examples

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
        generate_cdlm_embeddings(mention_map, vec_map_path, key_name='bert_doc', num_gpus=4, batch_size=20, cpu=False)
    # # read the vector_map pickle
    cdlm_vec_map = pickle.load(open(vec_map_path, 'rb'))
    # # generate and return the cosine similarities
    return get_cosine_similarities(mention_pairs, cdlm_vec_map)

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
         
        generate_cross_cdlm_embeddings(mention_pairs, mention_map, vec_map_path, key_name='bert_sentence', num_gpus=4, batch_size=150, cpu=False)
    # # read the vector_map pickle
    cdlm_scores = pickle.load(open(vec_map_path, 'rb'))
    
    return cdlm_scores

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

    within_doc_similarities = []

    doc_sent_map = pickle.load(open(working_folder + '/doc_sent_map.pkl', 'rb'))
    #doc_sims = pickle.load(open(working_folder + '/doc_sims_path.pkl', 'rb'))
    doc_ids = []

    for doc_id, _ in list(doc_sent_map.items()):
        doc_ids.append(doc_id)

    doc2id = {doc: i for i, doc in enumerate(doc_ids)}

    # generate similarity using the mention text
    for pair in mention_pairs:
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['mention_text'].lower()
        men_text2 = men_map2['mention_text'].lower()

        def jc(arr1, arr2):
            return len(set.intersection(arr1, arr2))/len(set.union(arr1, arr2))

        sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        #doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(men_map1['lemma'] in men_text2 or men_map2['lemma'] in men_text1)

        same_doc = float(men_map1['doc_id'] == men_map2['doc_id'])

        # similarities.append(sent_sim + doc_sim + lemma_sim)
        similarities.append((lemma_sim + 0.3*sent_sim)/2)
        # similarities.append((lemma_sim + 0.3*sent_sim)/2)
        within_doc_similarities.append(same_doc)


        # doc_plus_sent = 0. + doc_sim + sent_sim
        # if men_map1['lemma'] in men_text2 or men_map2['lemma'] in men_text1:
        #     # similarities.append(jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens'])))
        #     similarities.append(1. + doc_plus_sent)
        #
        #     if men_map1['doc_id'] == men_map2['doc_id']:
        #         within_doc_similarities.append(1 + sent_sim)
        #     else:
        #         within_doc_similarities.append(doc_sim)
        #
        # else:
        #     similarities.append(doc_sim + sent_sim)
        #     if men_map1['doc_id'] == men_map2['doc_id']:
        #         within_doc_similarities.append(sent_sim)
        #     else:
        #         within_doc_similarities.append(doc_sim)

    combined_sim = np.array(similarities) + np.array(within_doc_similarities)

    return np.array(similarities)


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


def coreference(curr_mention_map, all_mention_map, working_folder,
                men_type='evt', relations=None, sim_type='lemma',
                cluster_algo='cc', threshold=0.8, simulation=False,
                top_n=3):
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
    print(len(curr_mentions))
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
    
    
    print(len(mention_pairs))  
    # use generated pairs for triviality tests NOT NEEDED!
    
    # predict the labels for test pairs, separate the predicted trivial examples 
    
    
    # send trivial to lemma and get simialities, but map them to their IDS 
    
    
    #send the rest to cross encoder models and get scores i.e similarities,
    
    
    #stack the arrays and then do cc clustering and get final results. 
  
    
    
#     parent_path = '/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/'
#     triv_dev_path = parent_path + '/parsing/ecb/trivial_non_trivial_dev.csv'
#     triv_test_path = parent_path + '/parsing/ecb/trivial_non_trivial_test.csv'
#     with open("/s/chopin/d/proj/ramfis-aida/coref/coreference_and_annotations/coreference/test_indices", "rb") as fp:   
#         test_indices = pickle.load(fp)

#         test_pairs, test_labels = zip(*load_data_pair_coref(triv_test_path))

#         test_pairs = [test_pairs[i] for i in test_indices]
#         test_labels = [test_labels[i] for i in test_indices]
    
    
    
    # get the similarities of the mention-pairs from either lemmas or cdlm embeddings
    if sim_type == 'lemma':
        similarities = get_mention_pair_similarity_lemma(mention_pairs, all_mention_map, relations, working_folder)
         
    
    
#     if sim_type == 'lemma':  #for triviality pairs
#         similarities = get_mention_pair_similarity_lemma(test_pairs, all_mention_map, relations, working_folder)
#         print(len(similarities))
    elif sim_type == 'cdlm':
        similarities = get_mention_pair_similarity_cdlm_bi(mention_pairs, all_mention_map, relations, working_folder)
    elif sim_type == 'cross-encoder': 
        similarities = get_mention_pair_similarity_cdlm_cross(mention_pairs, curr_mention_map, relations, working_folder)
        similarities = [x for x in similarities.values()] #returns a dict with pairwise mention_ids as key, so only use scores 
    
    # get indices
    #maybe change it for trivialilty pairs
    mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in mention_pairs]
    #mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in test_pairs]
    rows, cols = zip(*mention_ind_pairs)
    print(len(rows), len(cols))

    # create similarity matrix from the similarities
    n = len(curr_mentions)
    
    #change current mentions for triviality pairs 
    #n = len(test_pairs)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    # clustering algorithm and mention cluster map
    if cluster_algo == 'cc':
        clusters, labels = cluster_cc(similarity_matrix)
    elif cluster_algo == 'agglo':
        clusters, labels = cluster_agglo(similarity_matrix)
    elif cluster_algo == 'inc':
        if not simulation:
            clusters, labels = incremental_clustering(similarity_matrix, threshold, curr_mentions,
                                                      all_mention_map, curr_men_to_ind, simulation=simulation,
                                                      top_n=top_n)
        else:
            clusters, labels, inc_clusterer = incremental_clustering(similarity_matrix, threshold,
                                                                          curr_mentions, all_mention_map,
                                                                          curr_men_to_ind, simulation=simulation,
                                                                          top_n=top_n)
    else:
        raise AssertionError
        
        
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)

    if simulation and cluster_algo == 'inc':
        return inc_clusterer


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

    # extract the mention maps
    mention_map, doc_sent_map = extract_mentions(ann_dir, source_dir, working_folder)

    curr_mention_map = {key: val for key, val in mention_map.items() if val['men_type'] == men_type}

    # do some filtering:
    curr_mention_map_new = {}
    for key, mention in curr_mention_map.items():
        mention_text = mention['mention_text']
        if len(mention_text.strip()) > 2 and len(mention_text.split()) < 4:
            curr_mention_map_new[key] = mention

    coreference(curr_mention_map_new, mention_map, working_folder, men_type, None,
                sim_type='lemma', cluster_algo='inc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run and evaluate cross-document coreference resolution on \
                                                 LDC Annotations')
    parser.add_argument('--ann', '-a', help='Path to the LDC Annotation Directory')
    parser.add_argument('--source', '-s', help='Path to the LDC Source Directory')
    parser.add_argument('--tmp_folder', '-t', default='./tmp', help='Path to a working directory')
    parser.add_argument('--men_type', '-m', default='evt', help='Mention type for coreference. Either evt or ent')
    args = parser.parse_args()
    print("Using the Annotation Directory: ", args.ann)
    print("Using the Source     Directory: ", args.source)
    print("Using the Working    Directory:", args.tmp_folder)
    run_coreference(args.ann, args.source, args.tmp_folder)
