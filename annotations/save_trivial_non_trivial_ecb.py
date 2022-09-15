from coreference.coref import coreference
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
    coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                cluster_algo='inc', threshold=0.1, simulation=simulation, top_n=3)


ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
working_folder = "../parsing/ecb"

run_coreference(ann_dir, working_folder)