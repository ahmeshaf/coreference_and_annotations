# Author: Rehan
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as thes
from nltk.corpus import framenet as fn
from annotations.action_words import get_lexeme_fn
from tqdm.autonotebook import tqdm


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


POS_MAP = {
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a',
}


def get_lexeme_spacy(token):
    """
    Create a lexeme from spacy Token of the form 'lemma.pos_'

    Parameters
    ----------
    token: spacy.tokens.Token

    Returns
    -------
    str
        A string of the form lemma.pos_ (e.g.: run.v, purchase.n, etc.)
    """

    return token.lemma_ + '.' + POS_MAP[token.pos_]


def get_lexeme2frames_fn():
    """
    Generate a map of lexeme (lemma.pos) to possible frames in frame net

    Returns
    -------

    """
    lex2frames = {}
    for lu in fn.lus():
        lexeme = get_lexeme_fn(lu)
        if lexeme not in lex2frames:
            lex2frames[lexeme] = []
        lex2frames[lexeme].append(lu.frame.name)
    return lex2frames


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
    for men_id, mention in tqdm(list(mention_map.items()), 'Adding Lexical Features'):
        # parse spacy over mention_text
        mention_nlp = nlp(mention['mention_text'])

        # mention_nlp_no_sw = nlp(' '.join([w.text for w in mention_nlp if not w.is_stop]))
        #
        # # if mention is a stop word
        # if mention_nlp_no_sw.text == '':
        #     mention['lemma'] = ""
        #     mention['frames'] = set()
        #     continue

        # get lemma
        mention['lemma'] = mention_nlp[:].root.lemma_

        # get POS
        mention['pos'] = mention_nlp[:].root.pos_

        # get framenet frames
        frames = set([f.name for f in fn.frames_by_lemma(mention['lemma'])])
        mention['frames'] = frames
        