from coref import coreference
from parsing.parse_ecb import parse_annotations
import pickle
import os


def run_coreference(ann_dir, working_folder, men_type='evt', split='dev',
                    algo='cc', sim='lemma', simulation=False, threshold=0.6):
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

    if not simulation:
        coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                    cluster_algo=algo, threshold=threshold, sim_type=sim, simulation=simulation, top_n=3)
    else:
        simulation_metrics = coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                                         cluster_algo='inc', threshold=-1, simulation=simulation, top_n=5)
        print(simulation_metrics)


def _generate_simulation_results_plot(men_type='evt', split='dev'):
    ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
    working_folder = "../parsing/ecb"
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

    if not simulation:
        coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                    cluster_algo='inc', threshold=0.4, simulation=simulation, top_n=3)
    else:
        top_ns = [3, 5, 10, 20]

        simulation_metrics_n = []

        for n in top_ns:
            simulation_metrics = coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                                             cluster_algo='inc', threshold=0.1, simulation=True, top_n=n)
            simulation_metrics_n.append(simulation_metrics)

        comparisons_plot(zip(top_ns, simulation_metrics_n))


def comparisons_plot(results):
    ns, sim_metrics = zip(*results)
    recalls, precisions, comparisons = zip(*sim_metrics)

    print(sim_metrics)

    import matplotlib.pyplot as plt

    plt.plot(comparisons, recalls, marker='x')
    plt.show()


def run_coref():
    ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
    working_folder = "../parsing/ecb"
    split = 'train'
    men_type = 'evt'
    clus_algo = 'cc'
    similarity = 'lemma'
    thres = 0.55
    run_coreference(ann_dir, working_folder, men_type, split, clus_algo, similarity, threshold=thres)


# _generate_simulation_results_plot()
run_coref()
