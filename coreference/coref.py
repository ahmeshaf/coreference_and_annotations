# Author: Rehan
# Run an end-to-end event coreference resolution experiment using the LDC annotations
import os
import sys
sys.path.insert(0, os.getcwd())

from parsing.parse_ldc import extract_mentions
import argparse
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from evaluations.eval import *
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')


class Cluster:
    def __init__(self, mention_dicts):
        self.mention_dicts = mention_dicts
        self.doc_ids = set([m['doc_id'] for m in mention_dicts])
        self.topic = mention_dicts[0]['topic']
        self.id_ = id(self)

    def merge_cluster(self, event):
        self.mention_dicts.extend(event.mention_dicts)
        self.doc_ids.update(event.doc_ids)

    def from_same_doc(self, event):
        return len(self.doc_ids.intersection(event.doc_ids)) > 0


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
        lemma1 = men_map1['lemma']
        lemma2 = men_map2['lemma']

        fnames1 = men_map1['frames']
        fnames2 = men_map2['frames']

        same_frame = len(fnames1.intersection(fnames2)) > 0

        similarities.append(int(lemma1 in men_text2 or lemma2 in men_text1))

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
            for j in range(i):
                mention_pairs.append((list_mentions[i], list_mentions[j]))

    # Generate pair-wise similarity matrix between the mentions of the same topic:
    # TODO: can be made into a func:
    #  generate_pair_wise_sim(mention_pairs, men2ind, mention_map, relations, working_folder)

    ## get the similarities of the mention-pairs
    similarities = get_mention_pair_similarity_lemma(mention_pairs, all_mention_map, relations, working_folder)
    ## get indices
    mention_ind_pairs = [(curr_men_to_ind[m1], curr_men_to_ind[m2]) for m1, m2 in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)
    ## create similarity matrix from the similarities
    n = len(curr_mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities
    similarity_matrix[cols, rows] = similarities

    # get within-doc mention clusters as {within_doc_cluster: [mention]}
    # TODO: do within-doc clustering
    within_doc_clusters = [Cluster([all_mention_map[mention]]) for mention in curr_mentions]

    # get cross-doc mention clusters.
    clusters, labels = cross_document_clustering(curr_mentions, within_doc_clusters, all_mention_map,
                                                 relations, similarity_matrix, curr_men_to_ind)

    # clustering algorithm and mention cluster map
    # clusters, labels = connected_components(similarity_matrix)
    system_mention_cluster_map = [(men, clus) for men, clus in zip(curr_mentions, labels)]

    # generate system key file
    system_key_file = working_folder + f'/{men_type}_system.keyfile'
    generate_key_file(system_mention_cluster_map, men_type, working_folder, system_key_file)

    # evaluate
    generate_results(gold_key_file, system_key_file)


def cross_document_clustering(mentions, within_doc_clusters, all_mention_map,
                              relations, similarity_matrix, curr_men_to_ind):
    """

    Parameters
    ----------
    mentions: list
    within_doc_clusters: list of Cluster
    all_mention_map: dict
    relations: list
    similarity_matrix: np.array
    curr_men_to_ind: dict

    Returns
    -------
    list, list
    """
    # group clusters by topic
    cluster_by_topic = defaultdict(list)
    for cluster in within_doc_clusters:
        cluster_by_topic[cluster.topic].append(cluster)

    # clustering clusters by topic
    mention_cluster_map = {}
    for topic, clusters in cluster_by_topic.items():
        cluster_sim_matrix = np.identity(len(clusters))
        for i, cluster_1 in enumerate(clusters):
            men_ids_1 = [curr_men_to_ind[m['mention_id']] for m in cluster_1.mention_dicts]
            for j, cluster_2 in enumerate(clusters):
                men_ids_2 = [curr_men_to_ind[m['mention_id']] for m in cluster_2.mention_dicts]
                max_sim = max([similarity_matrix[m1, m2] for m1, m2 in zip(men_ids_1, men_ids_2)])
                cluster_sim_matrix[i, j] = max_sim
        _, labels = cluster_cc(cluster_sim_matrix)
        for cluster, label in zip(clusters, labels):
            for mention in cluster.mention_dicts:
                mention_cluster_map[curr_men_to_ind[mention['mention_id']]] = topic + '_' + str(label)

    # create an ordered list of labels for mentions
    labels = [mention_cluster_map[men] for men in mentions]

    # clusters by labels
    mention_by_labels = defaultdict(list)
    for mention, label in mention_cluster_map.items():
        mention_by_labels[label].append(mention)

    return list(mention_by_labels.values()), labels


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
