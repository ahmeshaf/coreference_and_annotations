from collections import defaultdict, Counter
import random
random.seed(42)
from copy import copy


class IncrementalClusterer():
    """
        Class to do coreference resolution through incremental clustering.
        Class is useful to save states when running in annotation mode

        Class Members
        _____________
        event_clusters: dict
    """
    def __init__(self, filter_func=None):
        """

        Parameters
        ----------
        filter_func: func
            A function that does filtering of candidates (eg., based on topic)
        """
        self.event_clusters = defaultdict(set)
        self.lemma_clus_map = defaultdict(set)
        self.tok_clus_map = defaultdict(set)
        self.filter = filter_func

    def copy(self, inc_class):
        """
        Deep copy of the current class
        Parameters
        ----------
        inc_class: IncrementalClusterer

        Returns
        -------
        None
        """
        self.event_clusters = copy(inc_class.event_clusters)
        self.lemma_clus_map = copy(inc_class.lemma_clus_map)
        self.tok_clus_map = copy(inc_class.tok_clus_map)
        self.filter = inc_class.filter

    def clear_globals(self):
        """
        Clear global variables
        Returns
        -------
        None
        """
        self.event_clusters.clear()
        self.lemma_clus_map.clear()
        self.tok_clus_map.clear()

    def get_possible_clus_lemma(self, mentions_dicts):
        lemmas = set([m['lemma'] for m in mentions_dicts])
        possible_clus = set()
        for lemma in lemmas:
            possible_clus.update(self.lemma_clus_map[lemma])
        return possible_clus

    def get_possible_clus_from_toks(self, mentions_dicts):
        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        possible_clus = []
        for tok in all_tokens:
            cluses = self.tok_clus_map[tok]
            possible_clus.extend(cluses)
        return possible_clus

    def merge_cluster(self, clus, event, mentions_dicts):
        lemmas = set([m['lemma'] for m in mentions_dicts])

        for lemma in lemmas:
            self.lemma_clus_map[lemma].add(clus)
            if event in self.lemma_clus_map[lemma]:
                self.lemma_clus_map[lemma].remove(event)

        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        for tok in all_tokens:
            self.tok_clus_map[tok].add(clus)
            if event in self.tok_clus_map[tok]:
                self.tok_clus_map[tok].remove(event)

        self.event_clusters[clus].add(event)

    def add_cluster(self, eve, mentions_dicts):
        lemmas = set([m['lemma'] for m in mentions_dicts])
        for lemma in lemmas:
            self.lemma_clus_map[lemma].add(eve)

        all_tokens = [tok.lower() for mention in mentions_dicts for tok in mention['sentence_toks']]
        for tok in all_tokens:
            self.tok_clus_map[tok].add(eve)

        self.event_clusters[eve].add(eve)

    def from_same_doc(self, clus, event):
        events_in_cluster = self.event_clusters[clus]
        docs = set([e.split(':')[0] for e in events_in_cluster])
        eve_doc = event.split(':')[0]
        return eve_doc in docs

    def sample_candidates(self, event, mentions_dicts, token_threshold=1):
        possible_clus_lemma = list(self.get_possible_clus_lemma(mentions_dicts))
        possible_clus_tok = list(self.get_possible_clus_from_toks(mentions_dicts))

        tok_clus_counter = Counter(possible_clus_tok).most_common()
        tok_clus_thres = set([p[0] for p in tok_clus_counter if p[1] > token_threshold])

        possible_clus_tok = [p for p in possible_clus_tok if p in tok_clus_thres]
        possible_clus = [c for c in possible_clus_lemma + possible_clus_tok if not self.from_same_doc(c, event)]
        counter = Counter(possible_clus)
        possible_clus = [p[0] for p in counter.most_common() if event != p[0]]

        if self.filter != None:
            possible_clus = self.filter(event, possible_clus)

        return possible_clus

    def run_clustering_on(self, mentions, all_mention_map, working_folder, run_within_doc=True, simulation=True):
        """
        Run clustering algorithm:
            1. Group mentions by docs
            2. Run Within-doc clustering of mentions for each doc and create mention clusters
            3. Group mentions clusters by topic
            3. Run Cross-doc clustering
        Parameters
        ----------
        mentions: list
        all_mention_map: dict
        working_folder: str
        run_within_doc: bool
        simulation: bool

        Returns
        -------
        None
        """
        # 1. Group mentions by docs
        mentions_by_doc = defaultdict(list)
        for mention in mentions:
            mentions_by_doc[all_mention_map[mention]['doc_id']].append(mention)

        # 2. Run within-doc clustering
        if run_within_doc or ('within_doc_cluster' not in all_mention_map[mentions[0]]):
            add_within_doc_clusters(mentions_by_doc, all_mention_map, working_folder, simulation)


