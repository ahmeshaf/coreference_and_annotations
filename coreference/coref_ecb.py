from coreference.coref import coreference
from parsing.parse_ecb import parse_annotations
import pickle
import os
import torch


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


def _generate_simulation_results_plot(men_type='evt', split='test'):
    ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
    working_folder = "../parsing/gvc"
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
        top_ns = [i/2 for i in range(4, 21)]

        simulation_metrics_n = []
        sim_file = '../ecb_cdlm_scores/test_scores_all_pairs.pkl'
        similarities = torch.stack(pickle.load(open(sim_file, 'rb')))
        # similarities = torch.sigmoid(similarities)
        sig_similarities = torch.FloatTensor([torch.sigmoid(sim) for sim in similarities])
        # sig_similarities=None
        for n in top_ns:
            # similarities = torch.sigmoid(similarities)
            sim_cls = coreference(curr_mention_map, ecb_mention_map, working_folder, men_type=men_type,
                                             cluster_algo='inc', threshold=0.0001, simulation=True, top_n=n,
                                  sims=sig_similarities, sim_type='lemma')
            simulation_metrics_n.append(sim_cls.get_simulation_metrics())

        comparisons_plot(zip(top_ns, simulation_metrics_n))


def _run_simulation_experiment_sim_type(mention_map, ns, working_folder,
                                        sim_type, sim_file_path=None, threshold=0.0001, rand_run=False):
    if sim_file_path:
        similarities = torch.stack(pickle.load(open(sim_file_path, 'rb')))
        # similarities = torch.sigmoid(similarities)
        sig_similarities = torch.FloatTensor([torch.sigmoid(sim) for sim in similarities])
    else:
        sig_similarities = None

    simulation_metrics_no_thres = []
    simulation_metrics_thres = []

    print(f'Running ECB without threshold: {sim_type}')
    for n in ns:
        sim_cls = coreference(mention_map, mention_map, working_folder, men_type='evt',
                              cluster_algo='inc', threshold=-1, simulation=True, top_n=n,
                              sims=sig_similarities, sim_type=sim_type, rand_run=rand_run)
        simulation_metrics_no_thres.append(sim_cls.get_simulation_metrics())

    print(f'Running ECB with threshold: {sim_type}')
    for n in ns:
        sim_cls = coreference(mention_map, mention_map, working_folder, men_type='evt',
                              cluster_algo='inc', threshold=threshold, simulation=True, top_n=n,
                              sims=sig_similarities, sim_type=sim_type, rand_run=rand_run)
        simulation_metrics_thres.append(sim_cls.get_simulation_metrics())

    return simulation_metrics_thres, simulation_metrics_no_thres


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
    split = 'test'
    men_type = 'evt'
    clus_algo = 'cc'
    similarity = 'lemma'
    thres = 0.55
    run_coreference(ann_dir, working_folder, men_type, split, clus_algo, similarity, threshold=thres)


if __name__=='__main__':
    _generate_simulation_results_plot()

# run_coref()
