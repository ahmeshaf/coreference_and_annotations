from collections import defaultdict, Counter
import random
random.seed(42)
from copy import copy


class Cluster:
    """
    A class of cluster of mentions

    Attributes
    ----------
    mention_dicts: list of mention dicts
    doc_ids: set of doc ids of the mentions
    topic: str Topic ID
    gold_cluster: str cluster ID from annotations
    id_: id  unique ID for hashing the cluster
    """
    def __init__(self, mention_dict):
        self.mention_dicts = [mention_dict]
        self.doc_ids = set([m['doc_id'] for m in self.mention_dicts])
        self.topic = mention_dict['topic']
        self.gold_cluster = str(mention_dict['gold_cluster'])
        self.id_ = id(self)

    def merge_mention(self, mention_dict):
        self.mention_dicts.append(mention_dict)
        self.doc_ids.add(mention_dict['doc_id'])

    def merge_cluster(self, cluster):
        self.mention_dicts.extend(cluster.mention_dicts)
        self.doc_ids.update(cluster.doc_ids)

    def __hash__(self):
        return hash(self.id_)


class Clustering:
    """
    A class to do clustering. This class eases continuing incremental clustering
    upon exit

    Attributes
    __________
    mentions: list[str]
    lemma2clusters: dict
        A dict of str: set(Cluster)
    frame2clusters: dict
        A dict of str: set(Cluster)
    token2clusters: dict
        A dict of str: set(Cluster)
    clusters: list
        A list of current clusters
    comparisons: int

    """
    def __init__(self, mentions=None):
        if mentions is not None:
            self.mentions = mentions[:]
        self.lemma2clusters = defaultdict(set)
        self.frame2clusters = defaultdict(set)
        self.token2clusters = defaultdict(set)
        self.clusters = []
        self.comparisons = 0

    def load_state(self, other):
        """
        load the attributes from a pickled class
        Parameters
        ----------
        other: Clustering

        Returns
        -------
        None
        """
        self.mentions = other.mentions
        self.lemma2clusters = copy(other.lemma2clusters)
        self.frame2clusters = copy(other.frame2clusters)
        self.clusters = copy(other.clusters)
        self.comparisons = copy(other.comparisons)

    def candidates(self, target_mention):
        """
        Find a list of candidate Clusters for a target mention
        Parameters
        ----------
        target_mention: dict

        Returns
        -------
        list[Cluster]
        """
        topic = target_mention['topic']
        lemma = target_mention['lemma']
        # frames = target_mention['frames']
        sent_tokens = target_mention['sentence']

        candidates = set()
        candidates.update(self.lemma2clusters[lemma])
        # for frame in frames:
        #     candidates.update(self.frame2clusters[frame])
        for tok in sent_tokens:
            candidates.update(self.token2clusters[tok])

        return [cand for cand in candidates if cand.topic == topic]

    def nearest_neighbors(self, mention_dict, men2id, similarity_matrix, top_n=3):
        topic = mention_dict['topic']
        # get all topic clusters
        possible_clusters = [clus for clus in self.clusters if clus.topic == topic]

        men_id = mention_dict['mention_id']

        # sim values by taking the max sim between the target and mentions of candidate
        similarities = [
            max([similarity_matrix[men2id[men_id], men2id[m['mention_id']]] for m in clus.mention_dicts])
            for clus in possible_clusters
        ]

        zipped_clusters = zip(possible_clusters, similarities)

        return sorted(zipped_clusters, key=lambda x: -x[-1])[:top_n]

    def update_cluster_maps(self, cluster, mention_dict):
        lemma = mention_dict['lemma']
        sent_tokens = mention_dict['sentence_tokens']
        # frames = mention_dict['frames']
        frames = []
        self.lemma2clusters[lemma].add(cluster)
        for frame in frames:
            if frame is not None and frame != '':
                self.frame2clusters[frame].add(cluster)
        for tok in sent_tokens:
            self.token2clusters[tok].add(cluster)

    def merge_cluster(self, cluster, mention_dict):
        cluster.merge_mention(mention_dict)
        self.update_cluster_maps(cluster, mention_dict)

    def add_cluster(self, mention_dict):
        cluster = Cluster(mention_dict)
        self.clusters.append(cluster)
        self.update_cluster_maps(cluster, mention_dict)

    def run_clustering(self, mention_map, men2id, similarity_matrix,
                       top_n=100, threshold=0.1, simulation=False,
                       random_run=False):
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
            sorted_clusters = sorted_clusters[:top_n]

            # variable to know if the mention was merged /w existing cluster
            is_merged = False

            # iterate through the candidates, merge and break if suitable cluster is found
            for clus, sim in sorted_clusters:
                self.comparisons += 1
                # merge if 1. running simulation and gold cluster matches
                #       or 2. not running simulation and similarity is above threshold
                if (simulation and clus.gold_cluster == str(mention_dict['gold_cluster'])) or \
                        (not simulation and sim > threshold):
                    self.merge_cluster(clus, mention_dict)
                    is_merged = True
                    break

            if not is_merged:
                self.add_cluster(mention_dict)

        # return clusters, labels
        return [men2id[m['mention_id']] for clus in self.clusters for m in clus.mention_dicts],\
               {
            m['mention_id']: clus_id for clus_id, clus in enumerate(self.clusters) for m in clus.mention_dicts
        }


def incremental_clustering(similarity_matrix, threshold, mentions, mention_map, men2id):
    clustering = Clustering(mentions)
    clusters, mention_clus_map = clustering.run_clustering(mention_map, men2id, similarity_matrix,
                                                           threshold=threshold, simulation=False)
    # order the labels according to the mentions
    labels = [mention_clus_map[men] for men in mentions]

    return clusters, labels
