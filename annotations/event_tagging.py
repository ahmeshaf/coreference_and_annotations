from nltk.corpus import wordnet as wn
from nltk.corpus import propbank as pb
from tqdm.autonotebook import tqdm
from prodigy.components.loaders import JSONL, JSON
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string, set_hashes
from prodigy.components.printers import pretty_print_ner
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy import Language
from bs4 import BeautifulSoup
from typing import Iterable, Optional, List
from parsing.parse_ecb import parse_annotations
import prodigy
import pickle
import spacy
import glob
import copy
import os


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


def get_propbank_frames():
    """
    Get the dictionary of the predicate's lemma to the set of rolesets it refers to
    Use a probpank development folder to read the frames if provided.
    default: nltk's propbank

    Returns
    -------
    dict
        A dict of predicate's lemma to all its propbank frames
    """
    # separate verb and noun frames
    lemma2rolesets = {
        'v': {},
        'n': {}
    }

    for frame_file in tqdm(pb.fileids()[2:], desc='Building PB from nltk'):   # first two files are not frames
        frame_bs = BeautifulSoup(pb.open(frame_file).read(), features="lxml")
        predicates = frame_bs.find_all('predicate')      # if a frame file has > 1 predicates
        for predicate in predicates:
            lemma = predicate['lemma'].strip()
            if lemma not in lemma2rolesets['v']:
                lemma2rolesets['v'][lemma] = []
            for roleset in predicate.find_all('roleset'):
                # roleset is the same as sense
                lemma2rolesets['v'][lemma].append((roleset['id'].strip(), roleset['name'].strip()))

    return lemma2rolesets


def get_frames_from_framesets(frameset_folder):
    """
    Read the frameset files and create dictionaries of verbal and nominal predicates
    Parameters
    ----------
    frameset_folder: str
        A valid frameset folder

    Returns
    -------
    dict
    """
    # separate verb and noun frames
    lemma2rolesets = {
        'v': {},
        'n': {}
    }

    # go through the frame xml files in the frames directory
    if os.path.exists(frameset_folder + "/frames/"):
        for frame_file in tqdm(list(glob.glob(frameset_folder + "/frames/*.xml")), desc='Building FrameSet'):
            with open(frame_file) as ff:
                frame_bs = BeautifulSoup(ff.read(), features="lxml")
            predicates = frame_bs.find_all('predicate')  # if a frame file has > 1 predicates
            for predicate in predicates:
                for roleset in predicate.find_all('roleset'):
                    aliases = roleset.find_all('alias')
                    for alias in aliases:
                        pos_ = alias['pos']
                        if pos_ not in 'nv':
                            continue
                        lemma = alias.text.strip()
                        if lemma not in lemma2rolesets[pos_]:
                            lemma2rolesets[pos_][lemma] = []
                        # roleset is the same as sense
                        lemma2rolesets[pos_][lemma].append((roleset['id'].strip(), roleset['name'].strip()))
    return lemma2rolesets


def get_event_matcher(nlp: spacy.Language, event_lemma_dict: dict = None) -> Matcher:
    """
    Create a Span Matcher using provided Verb and Noun lemmas
    """

    if event_lemma_dict is None:
        event_lemma_dict = get_eventive_lemmas_wn()

    matcher = Matcher(nlp.vocab)

    # create verb and noun patterns
    patterns = [
        [{"POS": "VERB", "LEMMA": {"IN": list(event_lemma_dict['v'].keys())}}],
        [{"POS": "NOUN", "LEMMA": {"IN": list(event_lemma_dict['n'].keys())}}]
    ]

    lemma_key = 'EVENT_LEMMAS'
    matcher.add(lemma_key, patterns, greedy='FIRST')    # TODO: Which greedy is best?
    return matcher


@Language.factory("events", default_config={"lexicon": 'wordnet'})
def create_events_component(nlp: Language, name: str, lexicon: str):
    return EventRecognizer(nlp, lexicon)


class EventRecognizer:
    """
    Event Recognition class as a spacy pipeline
    """
    def __init__(self, nlp: Language, lexicon: str):
        if lexicon is None:
            lexicon = 'wordnet'
        self.lexicon = lexicon
        self.nlp = nlp
        self.event_lemma_dict = {'v': {}, 'n': {'purchase': []}}
        self.removed_lemmas = {'v': set(), 'n': set()}
        self.added_lemmas = {'v': set(), 'n': set()}
        if not (lexicon is None):
            if lexicon == 'wordnet':
                # wordnet events
                self.event_lemma_dict = get_eventive_lemmas_wn()
            elif lexicon == 'propbank':
                # propbank from nltk
                self.event_lemma_dict = get_propbank_frames()
            elif os.path.exists(lexicon):
                # FrameSet folder
                self.event_lemma_dict = get_frames_from_framesets(lexicon)
            else:
                print('Invalid lexicon. Using an empty matcher')

        # create the event matcher
        self.matcher = get_event_matcher(nlp, self.event_lemma_dict)

        # Register custom extension on the Doc for events
        if not Doc.has_extension("evts"):
            Doc.set_extension("evts", default=[])

    def __call__(self, doc: Doc) -> Doc:
        # add the matched spans and the frames/senses it correspond to
        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            if span.root.pos_ == 'VERB' or span.root.pos_ == 'NOUN':
                frames = []
                if span.root.lemma_ in self.event_lemma_dict:
                    frames = self.event_lemma_dict[span.root.lemma_]
                doc._.evts.append((span, frames))
        return doc

    def update(self, predicted_spans, annotated_spans):
        """
        Update the matcher
        Parameters
        ----------
        predicted_spans: list
        annotated_spans: list

        Returns
        -------
        None
        """
        false_positive_spans = set(predicted_spans) - set(annotated_spans)

        false_negative_spans = set(annotated_spans) - set(predicted_spans)

        pos_map = {"VERB": "v", "NOUN": "n"}

        # remove these
        for pos, lemma in false_positive_spans:
            if lemma in self.event_lemma_dict[pos_map[pos]]:
                self.event_lemma_dict[pos_map[pos]].pop(lemma)
                self.removed_lemmas[pos_map[pos]].add(lemma)

        # add these
        for pos, lemma in false_negative_spans:
            if lemma not in self.event_lemma_dict[pos_map[pos]]:
                self.event_lemma_dict[pos_map[pos]][lemma] = []
                self.added_lemmas[pos_map[pos]].add(lemma)

        self.matcher = get_event_matcher(self.nlp, self.event_lemma_dict)


def add_token_span(tokens, token_num, label):
    tokens_before = tokens[:token_num]
    char_offset = sum([len(tok['text']) + 1 if tok['ws'] else len(tok['text']) for tok in tokens_before])
    token = tokens[token_num]
    span = {
        "token_start": token_num,
        "token_end": token_num,
        "start": char_offset,
        "end": char_offset + len(token['text']),
        "text": token['text'],
        "label": label,
        # "disabled": True
    }
    return span


def make_tasks(nlp: Language, stream: Iterable):
    """
    Add a 'spans' key to each example, with predicted events.

    Returns
    -------
    Iterable[dict]
        generator for task dicts
    """
    texts = ((eg["text"], eg) for eg in stream)
    # make sure EventRecognizer is added to the nlp pipe
    for doc, eg in nlp.pipe(texts, as_tuples=True):
        # disable docs and sentence

        # eg['tokens'][0]['disabled'] = True
        # eg['tokens'][0]["style"] = {"color": "blue", "borderBottom": "4px solid red"}
        # eg['tokens'][2]['disabled'] = True
        # eg['tokens'][2]["style"] = {"color": "green", "borderBottom": "3px solid red"}
        task = copy.deepcopy(eg)
        # task_header = 'Hello' + '\n' + eg['doc_id'] + '\n' + str(eg['sentence_id']) + ' '
        # tok_offset = 2
        # char_offset = len(task_header)
        # task['text'] = task_header + eg['text']
        # eg['text'] = task['text']
        char_offset, tok_offset = (eg['char_offset'], eg['tok_offset'])
        # add document and sentence spans
        spans = []
        # spans.append(add_token_span(task['tokens'], 0, 'Document'))
        # spans.append(add_token_span(task['tokens'], 2, 'Sentence'))

        for evt in doc._.evts:
            evt_span, frames = evt
            spans.append(
                {
                    # "token_start": tok_offset + evt_span.start,
                    # "token_end": tok_offset + evt_span.end - 1,
                    # "start": char_offset + evt_span.start_char,
                    # "end": char_offset + evt_span.end_char,
                    "token_start": evt_span.start,
                    "token_end": evt_span.end - 1,
                    "start": evt_span.start_char,
                    "end": evt_span.end_char,
                    "text": evt_span.text,
                    "label": "EVT",
                }
            )

        task["spans"] = spans
        # Rehash the newly created task so that hashes reflect added data.
        task = set_hashes(task)
        yield task


count_additions, count_deletions = 0, 0


def read_docs(source_path):
    """
    Read docs represented as txt file where each line represents a sentence

    Parameters
    ----------
    source_path: str

    Returns
    -------
    list[dict]
    """
    for doc_txt in glob.glob(source_path + '/*.txt'):
        doc_id = os.path.basename(doc_txt)
        with open(doc_txt) as df:
            for i, sentence in enumerate(df):
                text = sentence.strip()
                header = "Document-" + doc_id + '\n' + "Sentence-" + str(i) + ' '
                yield {
                    'doc_id': doc_id,
                    'sentence_id': i,
                    'char_offset': len(header),
                    'tok_offset': 3,
                    # 'text': header + text,
                    'text': text,
                    'clean_text': text,
                    'meta': {
                        'Doc': doc_id,
                        'Sentence': i,
                        "Loc": os.path.abspath(doc_txt)
                    }
                }


@prodigy.recipe(
    "evt-tagging",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    lexicons=("Lexical resources to use for tagging", "option", "l", str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def eve_tagging_recipe(
        dataset: str,
        spacy_model: str,
        source: str,
        lexicons: Optional[str] = 'EMPTY',
        update: bool = False,
        exclude: Optional[List[str]] = None,
):
    """
    Create gold-standard data for events in text by updating model-in-the-loop
    """
    # Load the spaCy model.
    nlp = spacy.load(spacy_model)
    # add events into the pipeline
    nlp.add_pipe("events", config={'lexicon': lexicons})

    labels = ['EVT']

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('jsonl'):
        stream = JSON(source)
    elif os.path.isdir(source) and len(list(glob.glob(source + "/*.txt"))) > 0:
        stream = read_docs(source)
    else:
        stream = []

    stream = add_tokens(nlp, stream)
    stream = make_tasks(nlp, stream)

    def make_update(answers):
        global count_additions, count_deletions
        for answer in answers:
            # print(answer)
            nlp_doc = nlp(answer['clean_text'])
            print(nlp_doc)
            token_offset = answer['tok_offset']
            predicted_events = set([(span.root.pos_, span.root.lemma_) for (span, _) in nlp_doc._.evts])
            print(predicted_events)
            annotated_events = set()
            for span in answer['spans']:
                annotated_token = nlp_doc[span['token_start']: span['token_end'] + 1].root
                # annotated_token = nlp_doc[span['token_start']-token_offset: span['token_end']-token_offset + 1].root
                annotated_events.add((annotated_token.pos_, annotated_token.lemma_))
            print(annotated_events)
            if predicted_events != annotated_events:
                nlp.get_pipe("events").update(predicted_events, annotated_events)
                print('Updated Model')
            count_additions += len(set.difference(set(annotated_events), predicted_events))
            count_deletions += len(set.difference(set(predicted_events), annotated_events))

        return

    def on_exit(varsa):
        print("total additions:", count_additions)
        print("total deletions:", count_deletions)
        print("total changes:", count_additions + count_deletions)

    return {
        "view_id": "ner_manual",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_update if update else None,  # Update the model in the loop if required
        "exclude": exclude,  # List of dataset names to exclude
        "on_exit": on_exit,
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": labels,  # Selectable label options
            "span_labels": labels,  # Selectable label options
            "exclude_by": "input",  # Hash value to filter out seen examples
            "auto_count_stream": not update,  # Whether to recount the stream at initialization
            "batch_size": 1,
            "instant_submit": True
        },
    }


def _test_adaptive_event_tagging(ann_dir,
                                 working_folder,
                                 men_type='evt',
                                 split='dev',
                                 lexicon='EMPTY',
                                 spacy_model='en_core_web_md',
                                 heldoff_percent=0.1,
                                 adaptive=True):
    """

    Parameters
    ----------
    ann_dir
    working_folder
    men_type
    split
    lexicon,
    spacy_model,
    heldoff_percent
    adaptive

    Returns
    -------
    None
    """
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))

    nlp = spacy.load(spacy_model)
    nlp.add_pipe("events", config={'lexicon': lexicon})

    sentence_mention_map = {}
    sentence_text_map = {}
    mention_map = {eve: mention for eve, mention in ecb_mention_map.items() if
                   mention['men_type'] == men_type and mention['split'] == split}

    mention_map = {
        eve: mention for eve, mention in mention_map.items()
        if len(mention['mention_text'].split()) == 1 and
        len(mention['mention_text'].split('-')) == 1 and
        mention['pos'] in ['VERB', 'NOUN']
    }

    # group by topic, doc and ordered sentence ids
    """
    {
        topic:
        {
            doc_id:
            {
                sent_id: [men_id]
            }
        }
    }
    """

    for men_id, mention in mention_map.items():
        # sentence id
        topic, doc_id, sent_id = (mention['topic'], mention['doc_id'], int(mention['sentence_id']))
        sentence_id = (topic, doc_id, sent_id)

        # sentence text
        sentence_text_map[sentence_id] = mention['sentence']

        # sentence map
        if sentence_id not in sentence_mention_map:
            sentence_mention_map[sentence_id] = []
        sentence_mention_map[sentence_id].append(men_id)

    # group by topic and doc
    topic_doc2sentid = {}
    topic2docs = {}
    for sent_id in sentence_mention_map.keys():
        topic, doc, sent = sent_id
        if (topic, doc) not in topic_doc2sentid:
            topic_doc2sentid[topic, doc] = []
        topic_doc2sentid[topic, doc].append(sent_id)

        if topic not in topic2docs:
            topic2docs[topic] = set()
        topic2docs[topic].add(doc)

    # sort sent_ids in topic_doc2sentid for reproducibility
    for sent_ids in topic_doc2sentid.values():
        sent_ids.sort(key=lambda x: x[-1])

    # variables to calculate P, R, F1
    correct_predictions, total_predictions, gold_predictions = 0, 0, 0

    continue_using = True

    for topic, docs in tqdm(list(topic2docs.items()), desc='Running adaptive predictions'):

        if not continue_using:
            nlp.get_pipe('events').__init__(nlp, lexicon)

        # sort docs for reproducibility
        docs_sorted = sorted(list(docs))

        # held-out docs for adaptive learning
        train_docs = set(docs_sorted[:int(heldoff_percent * len(docs_sorted))])

        for doc in docs:
            for sent_id in topic_doc2sentid[topic, doc]:
                sentence_text = sentence_text_map[sent_id]
                sentence_mention_ids = sentence_mention_map[sent_id]
                gold_mentions = [mention_map[m_id] for m_id in sentence_mention_ids]

                # append gold_predictions for the sentence
                gold_predictions += len(gold_mentions)

                # get events
                nlp_sent = nlp(sentence_text)
                predicted_evts = nlp_sent._.evts

                # append total_predictions
                total_predictions += len(predicted_evts)

                def get_correct_predictions():
                    gold_start_spans = set([mention_map[m_id]['start_char'] for m_id in sentence_mention_ids])
                    gold_end_spans = set([mention_map[m_id]['end_char'] for m_id in sentence_mention_ids])

                    c_spans = set()
                    for evt_span, _ in predicted_evts:
                        if evt_span.start_char in gold_start_spans or evt_span.end_char in gold_end_spans:
                            c_spans.add(evt_span)

                    return c_spans

                correct_spans = get_correct_predictions()
                correct_predictions+=len(correct_spans)

                if adaptive and doc in train_docs:
                    predicted_events = [(span.root.pos_, span.root.lemma_) for (span, _) in predicted_evts]

                    annotated_events = set()
                    for men_id in sentence_mention_ids:
                        mention = mention_map[men_id]
                        annotated_token = nlp_sent[mention['start']: mention['end'] + 1].root
                        annotated_events.add((annotated_token.pos_, annotated_token.lemma_))
                    nlp.get_pipe("events").update(predicted_events, annotated_events)

    if adaptive:
        print(lexicon, '- Adaptive')
    else:
        print(lexicon, '- Non-Adaptive')
    print('Precision', correct_predictions/total_predictions)
    print('Recall', correct_predictions/gold_predictions)
    print('F1', 2*correct_predictions/(total_predictions + gold_predictions))
    with open('debug.txt', 'w') as df:
        df.write('------REMOVED-------\n')
        df.write('\n'.join([str(ite) for ite in nlp.get_pipe("events").removed_lemmas.items()]))

        df.write('\n------ADDED-------\n')
        df.write('\n'.join([str(ite) for ite in nlp.get_pipe("events").added_lemmas.items()]))


if __name__ == '__main__':
    # dataset = 'eve_coref'
    # # spacy_model = 'en_core_web_sm'
    # source = '/Users/rehan/workspace/prodigy/data/eve_coref.jsonl'
    #
    # # # tasks = eve_tagging_recipe(dataset, spacy_model, source, lexicons='wordnet')
    # pb_dev_folder = '/Users/rehan/workspace/propbank-development-master/'
    # # tasks = eve_tagging_recipe(dataset, spacy_model, source, lexicons=pb_dev_folder)
    # # for task_ in tasks['stream']:
    # #     pretty_print_ner([task_])
    # ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
    # working_folder = "../parsing/ecb"
    # men_type = 'evt'
    # split = 'train'
    # lexicon = 'EMPTY'
    # spacy_model = 'en_core_web_md'
    # heldoff_percent = 1.0
    # adaptive = True
    # _test_adaptive_event_tagging(ann_dir, working_folder, men_type, split,
    #                              lexicon, spacy_model, heldoff_percent, adaptive)

    source = '/Users/rehan/workspace/prodigy/data/docs/'
    lexicon = 'EMPTY'
    spacy_model = 'en_core_web_sm'
    something = eve_tagging_recipe(
        'evts1',
        spacy_model,
        source,
        lexicon)

    for eg in something['stream']:
        pretty_print_ner([eg])
