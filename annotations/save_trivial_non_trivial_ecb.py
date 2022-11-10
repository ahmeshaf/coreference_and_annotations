from coreference.coref import coreference, get_mention_pair_similarity_lemma
from parsing.parse_ecb import parse_annotations
import pickle
import os


def run_coreference(ann_dir, working_folder, men_type='evt', split='train'):
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

    simulation = True
    inc_clusterer = coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                                cluster_algo='inc', threshold=0.0, simulation=simulation, top_n=3)

    with open(working_folder + f'/trivial_non_trivial_{split}.csv', 'w') as tnf:
        tnf.write(
            '\n'.join([
                ','.join(row) for row in inc_clusterer.trivial_non_trivial
            ])
        )


def save_easy_hard(dataset, men_type='evt', split='train', threshold=0.2):
    from collections import defaultdict
    import numpy as np

    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == men_type
                        and val['split'] == split
                        # and len(val['mention_text'].split()) == 1
                        }

    single_word_eve = [eve for eve, val in curr_mention_map.items() if len(val['mention_text'].split()) == 1]
    multi_word_eve = [eve for eve, val in curr_mention_map.items() if len(val['mention_text'].split()) > 1]

    with open('test_mention.txt', 'w') as tff:
        tff.write('\n'.join([curr_mention_map[eve]['mention_text'] for eve in multi_word_eve]))

    print('total eves:', len(curr_mention_map))
    print('single word eves:', len(single_word_eve))
    print('multi word eves:', len(multi_word_eve))

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

    # similarities = np.array([0]*len(mention_pairs))
    similarities = np.array(get_mention_pair_similarity_lemma(mention_pairs, ecb_mention_map, [], working_folder))

    print(np.min(similarities))

    ground_truth = np.array([ecb_mention_map[m1]['gold_cluster'] == ecb_mention_map[m2]['gold_cluster'] for m1, m2 in mention_pairs])

    lemma_coref = similarities > threshold

    print('all positives:', lemma_coref.sum())
    print('true positives:', np.logical_and(lemma_coref, ground_truth).sum())
    print('false positives:', np.logical_and(lemma_coref, np.logical_not(ground_truth)).sum())
    print('true negatives:', np.logical_and(np.logical_not(lemma_coref), np.logical_not(ground_truth)).sum())
    print('false negatives:', np.logical_and(np.logical_not(lemma_coref), ground_truth).sum())

    false_positives = np.where(np.logical_and(np.logical_not(lemma_coref), ground_truth))

    fp_men_pairs = [mention_pairs[i] for i in false_positives[0]]

    with open(working_folder + f'/lemma_balanced_tp_fp_{split}.tsv', 'w') as tpf:
        tps = np.where(np.logical_and(lemma_coref, ground_truth))
        fps = np.where(np.logical_and(lemma_coref, np.logical_not(ground_truth)))
        tps_pairs = [mention_pairs[i] for i in tps[0]]
        fps_pairs = [mention_pairs[i] for i in fps[0]]
        tpf.write(
            '\n'.join(['\t'.join([m1, m2, 'POS']) for m1, m2 in tps_pairs])
        )
        tpf.write('\n')
        tpf.write(
            '\n'.join(['\t'.join([m1, m2, 'NEG']) for m1, m2 in fps_pairs])
        )

    print(len(similarities))


ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
working_folder = "../parsing/ecb"

save_easy_hard(None,split='dev')