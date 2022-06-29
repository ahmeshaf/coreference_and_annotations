# Author: Rehan
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as thes
from nltk.corpus import framenet as fn


def get_derivationally_related_verbs(spacy_doc):
    """
    Get all the derivationally related verbs for nouns
    Parameters
    ----------
    spacy_doc: spacy.doc

    Returns
    -------
    list: A list of string
    """
    word = spacy_doc.text
    lemma_set = set()

    # get spacy_token
    spacy_token = spacy_doc[0]

    # get derivational verb only if lemma is not a verb
    if spacy_token.pos_ != 'VERB':
        non_verb_synsets = wn.synsets(word)
        if len(non_verb_synsets) > 0:
            # choose the most popular synset according to wordnet
            syn = non_verb_synsets[0]
            lemmas = syn.lemmas()
            for lemma in lemmas:
                # get all the derivationally related form of the word
                deriv_related_forms = lemma.derivationally_related_forms()
                for form in deriv_related_forms:
                    # add only if the derivational form is a verb
                    if form.synset().pos() == 'v':
                        lemma_set.add(form._name)
    return list(lemma_set)


def add_sentential_features(nlp, mention_map):
    for mention in mention_map.values():
        sent_nlp = nlp(mention['sentence'])
        mention['sentence_tokens'] = []
        for token in sent_nlp:
            if not token.is_stop:
                mention['sentence_tokens'].append(token.lemma_)


def add_lexical_features(nlp, mention_map):
    """
    Add lemma, derivational verb, etc
    Parameters
    ----------
    nlp: spacy.tokens.Language
    mention_map: dict

    Returns
    -------
    None
    """
    for men_id, mention in mention_map.items():
        # parse spacy over mention_text
        mention_nlp = nlp(mention['mention_text'])
        # # default lemma if mention_text is just one token excluding stop word
        # lemma = ' '.join([t.lemma_ for t in mention_nlp if not t.is_punct and not t.is_punct])
        # lemma_vector = mention_nlp.vector
        # # when there are multiple tokens in mention_text
        # for tok in mention_nlp:
        #     if not tok.is_stop and not tok.is_punct:
        #         lemma = tok.head.lemma_
        #         lemma_vector = tok.vector
        mention['lemma'] = mention_nlp[:].root.lemma_

        frames = set([f.name for f in fn.frames_by_lemma(mention['lemma'])])
        mention['frames'] = frames

        # mention['lemma_vector'] = lemma_vector

        # add derivational verbs
        # mention['derivational_verbs'] = []
        # if lemma.strip() != '':
        #     mention['derivational_verbs'] = get_derivationally_related_verbs(nlp(lemma))

