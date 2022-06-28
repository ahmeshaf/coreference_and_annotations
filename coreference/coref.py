# Author: Rehan
# Run an end-to-end event coreference resolution experiment using the LDC annotations
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())
from parsing.parse_ldc import extract_mentions
from bert_stuff import *
import argparse

from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from evaluations.eval import *
import numpy as np


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
        return a / np.linalg.norm(a, axis=1).reshape((-1, 1))

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
    predictions = clustering_.fit_predict(distance_matrix)
    return predictions


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
                men_type='evt', relations=None, sim_type='lemma'):
    """

    Parameters
    ----------
    curr_mention_map
    all_mention_map
    working_folder
    men_type
    sim_type
    relations

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

    # get the similarities of the mention-pairs from either lemmas or cdlm embeddings
    if sim_type == 'lemma':
        similarities = get_mention_pair_similarity_lemma(mention_pairs, all_mention_map, relations, working_folder)
    elif sim_type == 'cdlm':
        similarities = get_mention_pair_similarity_cdlm_bi(mention_pairs, all_mention_map, relations, working_folder)

    # get indices
    mention_ind_pairs = [(curr_men_to_ind[mp[0]], curr_men_to_ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(curr_mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    # clustering algorithm and mention cluster map
    clusters, labels = connected_components(similarity_matrix)
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)


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
    eve_mention_map, ent_mention_map, relations, doc_sent_map = extract_mentions(ann_dir, source_dir, working_folder)

    # which coreference mention map
    if men_type == 'evt':
        curr_mention_map = eve_mention_map
    else:
        curr_mention_map = ent_mention_map

    # create a single dict for all mentions
    all_mention_map = {**eve_mention_map, **ent_mention_map}

    coreference(curr_mention_map, all_mention_map, working_folder, men_type, relations, sim_type='cdlm')


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