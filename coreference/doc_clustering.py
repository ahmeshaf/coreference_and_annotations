import os
import spacy
import pickle
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import norm
from collections import defaultdict
from spacy.matcher import Matcher


def get_eventive_lemmas_wn():
    """
    Get the dict of eventive noun/verb lemmas and their
    word senses using the WordNet lexicon

    Returns
    -------
    dict
         A dict of predicate's lemma to all its WN senses
    """
    # a set of eventive noun classes
    eventive_noun_classes = {
        'act', 'event', 'communication', 'state', 'cognition', 'feeling', 'phenomenon'
    }

    lemma2_synsets = {}
    for syn in tqdm(wn.all_eng_synsets(), total=117659, desc="Building WN Events"):
        syn_name: str = syn.name()  # syn set name. eg. battle.n.01
        syn_pos = syn.pos()    # part of speech ['v', 'n', etc]
        lex_name = syn.lexname().split('.')[-1]     # WN class. eg. noun.act

        # Add all verb senses.
        # For nouns, check the noun class
        if  syn_pos == 'v' or \
            (syn_pos == 'n' and
                lex_name in eventive_noun_classes):
            for lemma in syn.lemma_names():
                if syn_pos not in lemma2_synsets:
                    lemma2_synsets[syn_pos] = {}
                if lemma not in lemma2_synsets[syn_pos]:
                    lemma2_synsets[syn_pos][lemma] = set()
                lemma2_synsets[syn_pos][lemma].add(syn_name)
    return lemma2_synsets


def preprocess_docs_event_entity(docs_sentences, nlp, working_folder, force_preprocess=False):
    """
    Extract lemmas corresponding to events and named entities in the text

    Parameters
    ----------
    docs_sentences
    nlp: spacy.Language
    working_folder
    force_preprocess

    Returns
    -------

    """
    preprocessed_docs_path = working_folder + '/preprocessed_docs.pkl'
    if os.path.exists(preprocessed_docs_path) and not force_preprocess:
        return pickle.load(open(preprocessed_docs_path, 'rb'))

    matcher = Matcher(nlp.vocab)

    # create pattern for events
    event_lemma_dict = get_eventive_lemmas_wn()
    # create verb and noun patterns
    patterns = [
        [{"POS": "VERB", "LEMMA": {"IN": list(event_lemma_dict['v'].keys())}}],
        [{"POS": "NOUN", "LEMMA": {"IN": list(event_lemma_dict['n'].keys())}}]
    ]
    lemma_key = 'EVENT_LEMMAS'
    matcher.add(lemma_key, patterns, greedy='FIRST')

    # create pattern for Proper Nouns
    matcher.add('PROPN', [[{'POS': 'PROPN'}]], greedy='LONGEST')

    preprocessed_docs = [
        [' '.join([sent_doc[start].lemma_ for _, start, _ in matcher(sent_doc)])
            for sent_doc in nlp.pipe(sentences)]
        for sentences in tqdm(docs_sentences, desc='Extracting events and entities')
    ]

    preprocessed_docs = ['\n'.join(sentences) for sentences in preprocessed_docs]
    pickle.dump(preprocessed_docs, open(preprocessed_docs_path, 'wb'))
    return preprocessed_docs


def get_doc_distance_matrix(doc_sentences, nlp, working_folder, ngram_range=None, force_preprocess=False):
    """
    Generate the affinity matrix of the documents in tfidf form of events and proper nouns

    Parameters
    ----------
    doc_sentences: list[list[str]]
    nlp: spacy.Language
    working_folder: str
    ngram_range: tuple
    force_preprocess: bool

    Returns
    -------
    np.array
        A matrix of pairwise similarity values of the documents
    """
    preprocessed_docs = preprocess_docs_event_entity(doc_sentences,
                                                     nlp,
                                                     working_folder,
                                                     force_preprocess=force_preprocess)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectorized_docs = csc_matrix(tfidf_vectorizer.fit_transform(preprocessed_docs))
    vectorized_docs_norm_inv = 1 / norm(vectorized_docs, axis=1).reshape((-1, 1))
    vectorized_docs_normed = csc_matrix(vectorized_docs).multiply(vectorized_docs_norm_inv).tocsr()
    doc_cosine_dists = 1 - vectorized_docs_normed.dot(vectorized_docs_normed.T).toarray()
    return doc_cosine_dists


def cluster(doc_sent_map: dict, nlp: spacy.Language,
            threshold=0.9, linkage='average', working_folder='./tmp',
            ngram_range: tuple = None, force_preprocess: bool = False) -> np.array:
    """
    Agglomeratively cluster Tfidf representations of the docs
    Returns an array of labels corresponding to the doc's cluster
    """

    doc_ids, sent_maps = (zip(*list(doc_sent_map.items())))
    doc_sentences = [[sm['sentence'] for sm in sent_map.values()] for sent_map in sent_maps]

    doc_dists_path = working_folder + '/doc_sims_path.pkl'
    if not os.path.exists(doc_dists_path):
        doc_cosine_dists = get_doc_distance_matrix(doc_sentences, nlp, working_folder,
                                                   ngram_range=ngram_range, force_preprocess=force_preprocess)
        pickle.dump(doc_cosine_dists, open(doc_dists_path, 'wb'))

    doc_cosine_dists = pickle.load(open(doc_dists_path, 'rb'))

    clustering = AgglomerativeClustering(n_clusters=None,
                                         affinity='precomputed',
                                         distance_threshold=threshold,
                                         linkage=linkage)

    labels = clustering.fit_predict(doc_cosine_dists)

    doc_cluster_map = {doc_id: label for doc_id, label in zip(doc_ids, labels)}

    return doc_cluster_map


def get_entity_matrix(mentions, doc_cluster_map):
    mention_ids = sorted(mentions.keys())
    men2ind = {men: i for i, men in enumerate(mention_ids)}

    n = len(mention_ids)

    coref_matrix = lil_matrix(np.identity(n))

    gold_clus_dict = defaultdict(list)

    for m_id, mention in mentions.items():
        gold_clus_dict[mention['gold_cluster']].append(m_id)

    for clusters_ in gold_clus_dict.values():
        for m_id1 in clusters_:
            doc_id1 = mentions[m_id1]['doc_id']
            for m_id2 in clusters_:
                doc_id2 = mentions[m_id2]['doc_id']
                if doc_cluster_map[doc_id1] == doc_cluster_map[doc_id2]:
                    coref_matrix[men2ind[m_id1], men2ind[m_id2]] = 1

    # for i, m_id1 in enumerate(tqdm(mention_ids, desc='preparing coref matrix')):
    #     mention1 = mentions[m_id1]
    #     for j, m_id2 in enumerate(mention_ids):
    #         mention2 = mentions[m_id2]
    #         if doc_cluster_map[mention1['doc_id']] == doc_cluster_map[mention2['doc_id']] and \
    #                 mention1['men_type'] == mention2['men_type']:
    #             coref_matrix[i, j] = mentions[m_id1]['gold_cluster'] == mentions[m_id2]['gold_cluster']

    return coref_matrix


def get_doc_comparisons(doc_map):
    doc_clusters = defaultdict(list)
    for doc, clus in doc_map.items():
        doc_clusters[clus].append(doc)

    comps = sum([(len(clus)*(len(clus) - 1))//2 for clus in doc_clusters.values()])
    return comps


if __name__ == '__main__':
    sp_nlp = spacy.load('en_core_web_sm')

    wf = "../parsing/ecb"

    # compare_annotations(sp_nlp, wf)




