from wasabi import color, wrap
import os
import sys
sys.path.insert(0, os.getcwd())
from parsing.parse_ldc import extract_mentions
# from bert_stuff import generate_cdlm_embeddings
import argparse
from evaluations.eval import *
from collections import defaultdict
import numpy as np
from coreference.incremental_clustering import Clustering
import pickle


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


def add_formatted_sentence(mention_map, relations=None):
    """
    Add color coded sentence using the bert sentence
    Parameters
    ----------
    mention_map: dict
    relations: list

    Returns
    -------
    None
    """
    relation_map_head = {}
    for (head, relation, tail) in relations:
        if head not in relation_map_head:
            relation_map_head[head] = []
        relation_map_head[head].append([relation[6:11].upper(), tail])

    for mention_id, mention in mention_map.items():
        bert_sentence = mention['bert_sentence']
        bert_sent_tokens = bert_sentence.split()

        m_start = bert_sent_tokens.index('<m>')
        m_end = bert_sent_tokens.index('</m>')

        # split tokens
        left_part = ' '.join(bert_sent_tokens[:m_start])
        mention_part = ' '.join(bert_sent_tokens[m_start+1: m_end])
        right_part = ' '.join(bert_sent_tokens[m_end+1:])

        formatted_sentence = left_part + " " + \
                                color(f" {mention_part} ", fg="black", bg=220, bold=True) + \
                                color(f" {mention['men_type'].upper()} ", fg="black", bg=81, bold=True) + \
                                " " + right_part

        mention['formatted_sentence'] = formatted_sentence

        mention['spacy_span'] = [{
            'text': mention['sentence'],
            'spans': [{'start': mention['start_char'] - mention['sentence_start_char'],
                       'end': mention['end_char'] - mention['sentence_start_char'] + 1,
                       'label': mention['men_type'].upper()}]
        }]

        if mention_id in relation_map_head:
            head_relations = relation_map_head[mention_id]
            for (label, tail) in head_relations:
                if tail in mention_map:
                    tail_mention = mention_map[tail]
                    if tail_mention['sent_id'] == mention['sent_id']:
                        mention['spacy_span'][0]['spans'].append({
                            'start': tail_mention['start_char'] - tail_mention['sentence_start_char'],
                            'end': tail_mention['end_char'] - tail_mention['sentence_start_char'] + 1,
                            'label': label
                        })
            mention['spacy_span'][0]['spans'] = sorted(mention['spacy_span'][0]['spans'], key=lambda x: x['start'])
        pass


def pretty_print_spans(spacy_spans):
    text = spacy_spans['text']
    spans = spacy_spans['spans']

    target_bg = 81
    target_label_color_fg = 220
    targets = {'EVT', 'ARG'}
    args_color = 159

    spans_sorted = sorted(spans, key=lambda x: x['start'])

    pretty_spans = [
        color(f" {text[s['start']: s['end']]} ", fg='black', bg=220 if s['label'] in targets else 230, bold=False) +
        color(f" {s['label']} ", fg='black', bg=81 if s['label'] in targets else 227, bold=True)
        for s in spans_sorted
    ]

    zipped_span = list(zip(spans_sorted, pretty_spans))

    index = 0
    pretty_text = ""

    while index < len(text):
        if len(zipped_span) > 0:
            (curr_span, curr_pretty_span) = zipped_span.pop(0)

            span_start, span_end = curr_span['start'], curr_span['end']

            # when a span label comes later on
            if index <= span_start:
                pretty_text += text[index: span_start] + curr_pretty_span
                index = span_end
            elif span_start < index < span_end:
                # when there is an overlapping span
                pretty_text += curr_pretty_span
                index = span_end
            else:
                # when there is a fully contained span in previous span
                pretty_text += curr_pretty_span
        else:
            pretty_text += text[index:]
            index = len(text)

    return pretty_text.strip()


def pretty_print_cluster(cluster, clus_num):
    clus_num = 2 # TODO: change this back
    print("\n", color(f" Candidate: {clus_num} ", fg="black", bg=10))
    for i, mention in enumerate(cluster.mention_dicts):
        print("\n",
              color(f" C {clus_num}.{i}: ", fg="black", bg=10) +
              color(f" {mention['doc_id']} ", fg=123, bg=22),
              " " + pretty_print_spans(mention['spacy_span'][0]), '\t'*100, "\n")


def pretty_print_mention(mention):
    print("\n", color(" Target Mention ", fg="black", bg=81))
    print("\n", color(" T: ", fg="black", bg=81) +
                color(f" {mention['doc_id']} ", fg=123, bg=22),
          " " + pretty_print_spans(mention['spacy_span'][0]), '\t'*200)


class HumanClustering(Clustering):
    """
    A class to run Human-in-the-loop Clustering for Annotating coreference relations
    """
    def __init__(self, mentions, working_folder):
        super().__init__(mentions)
        self.working_folder = working_folder

    def run_clustering(self, mention_map, men2id, similarity_matrix,
                       top_n=100, threshold=0.8, simulation=False,
                       random_run=False):
        true_positives = 0
        false_negatives = 0
        total_comps = 0
        while len(self.mentions) > 0:
            mention = self.mentions.pop(0)
            mention_dict = mention_map[mention]
            cluster_candidates = self.candidates(mention_dict)

            similarities = [
                max([similarity_matrix[men2id[mention], men2id[m['mention_id']]] for m in clus.mention_dicts])
                for clus in cluster_candidates
            ]

            # zip candidate cluster with the similarity value
            zipped_clusters = zip(cluster_candidates, similarities)

            # sort clusters based on similarity
            sorted_clusters = sorted(zipped_clusters, key=lambda x: -x[-1])

            # ignore sorting by scores - case when we want to test fully manual
            # random simulation
            if random_run:
                sorted_clusters = zipped_clusters

            # pick top-n clusters
            # sorted_clusters = sorted_clusters[:top_n]

            # variable to know if the mention was merged /w existing cluster
            is_merged = False

            if len(sorted_clusters) > 0:
                total_comps += 1

            # iterate through the candidates, merge and break if suitable cluster is found
            for i, (clus, sim) in enumerate(sorted_clusters):
                if i >= top_n:
                    break
                self.comparisons += 1

                # TODO: Remove this
                # if clus.gold_cluster == str(mention_dict['gold_cluster']) and 'Mel' in mention_dict['sentence']:
                #     print("\n ----- Question - Coreferent? -----")
                #     # pretty_print_cluster(clus, i)
                #     # pretty_print_mention(mention_dict)
                #
                #     pretty_print_cluster(clus, i)
                #     pretty_print_mention(mention_dict)
                #
                #     answer = input("\n  Answer: y/n?")
                #     # print(" Answer: y/n? ")

                if clus.gold_cluster == str(mention_dict['gold_cluster']):
                    # print target mention and its candidates
                    # print("\n ----- Question - Coreferent? -----")
                    # pretty_print_cluster(clus, i)
                    # pretty_print_mention(mention_dict)
                    # print(" Answer: y/n? ")
                    # answer = input("  Answer: y/n?")
                    # print(" ------ Question - End ------")
                    true_positives += 1
                    self.merge_cluster(clus, mention_dict)
                    is_merged = True
                    break

            if not is_merged:
                if mention_dict['gold_cluster'] in set([clus.gold_cluster for clus, _ in sorted_clusters]):
                    false_negatives += 1
                self.add_cluster(mention_dict)

        print('Recall: ', true_positives/(true_positives + false_negatives))
        # return clusters, labels
        return [men2id[m['mention_id']] for clus in self.clusters for m in clus.mention_dicts],\
               {
            m['mention_id']: clus_id for clus_id, clus in enumerate(self.clusters) for m in clus.mention_dicts
        }


def coreference(curr_mention_map, all_mention_map, working_folder,
                men_type='evt', relations=None, sim_type='lemma',
                threshold=0.8, top_n=5):
    """

    Parameters
    ----------
    curr_mention_map
    all_mention_map
    working_folder
    men_type
    relations
    sim_type
    threshold: double
    Returns
    -------

    """

    # sort event mentions and make men to ind map
    # curr_mentions = sorted(list(curr_mention_map.keys()), key=lambda x: curr_mention_map[x]['doc_id'])
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

    clustering = HumanClustering(curr_mentions, working_folder)
    clusters, mention_clus_map = clustering.run_clustering(all_mention_map, curr_men_to_ind, similarity_matrix,
                                                           threshold=threshold, simulation=False, top_n=top_n)

    # order the labels according to the mentions
    labels = [mention_clus_map[men] for men in curr_mentions]

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

    # do some filtering:
    curr_mention_map_new = {}
    for key, mention in curr_mention_map.items():
        mention_text = mention['mention_text']
        if len(mention_text.strip()) > 2 and len(mention_text.split()) < 4:
            curr_mention_map_new[key] = mention

    # create a single dict for all mentions
    all_mention_map = {**eve_mention_map, **ent_mention_map}

    add_formatted_sentence(all_mention_map, relations)

    coreference(curr_mention_map_new, all_mention_map, working_folder, men_type, relations,
                sim_type='lemma', top_n=5)


from parsing.parse_ecb import parse_annotations

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
    # run_coreference(args.ann, args.source, args.tmp_folder, men_type=args.men_type)

    def run_coreference(ann_dir, working_folder, men_type='evt', split='test'):
        """

        Parameters
        ----------
        ann_dir
        working_folder
        men_type
        split

        Returns
        -------

        """
        # read annotations
        ecb_mention_map_path = working_folder + '/mention_map.pkl'
        if not os.path.exists(ecb_mention_map_path):
            parse_annotations(ann_dir, working_folder)
        ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))

        for key, val in ecb_mention_map.items():
            val['mention_id'] = key

        curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == men_type and
                            val['split'] == split}

        # create a single dict for all mentions

        # add_formatted_sentence(all_mention_map, relations)

        coreference(curr_mention_map, ecb_mention_map, working_folder, men_type, [],
                    sim_type='lemma')


    ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
    working_folder = "../parsing/ecb"

    run_coreference(ann_dir, working_folder)

