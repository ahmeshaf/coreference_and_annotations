# Author: Abhijnan Nath
# Parse the GVC Gold Annotations and GVC verbose files to generate a common representation of events, sentences that contains those events and their respective document IDS

import sys 
import os
import argparse
sys.path.insert(0, os.getcwd())
#sys.path.append("/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/")
sys.path.append("/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/")
# #sys.path.append('.')
sys.argv = ['']
import json
import requests
import copy
import pandas as pd
import time
import unicodeblock.blocks
from tqdm import tqdm
from random import random
import numpy as np
import pickle
from io import open
from pathlib import Path
import re
from collections import OrderedDict, defaultdict

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token= line.split('\t')
            tokens.append(token)
             
        token_docs.append(tokens)
       

    return token_docs, tokens



def create_gold_metaraw(gold_filepath, meta_filepath):
    
    
    """
    Create correctly mapped multiple-token events for GVC corpus'

    Parameters
    ----------
    gold_filepath: Path to the GVC Gold Annotation Directory
    meta_filepath : Path to the GVC Verbose(meta data for event type) Annotation Directory
 
    Returns
    -------
    gold_df_retransposed_list : all correctly mapped mentions after combining multi-token events as 
    a single event with meta data as Pandas Dataframe
    
    meta_df_retransposed_list : 
    
    mention_csv_all : all correctly identitied mentions with meta data as Pandas Dataframe
    """
    c=0  
    all_mentions = []  #finding the combinations of bracketted events 
    two_combo_idx = []
    three_combo_idx = []
    four_combo_idx = []
    five_combo_idx = []
    six_combo_idx = []
    eight_combo_idx = []
    elev_combo_idx = []

    
    
    gold_texts, gold_tok = read_wnut(gold_filepath)
    meta_texts, meta_tok = read_wnut(meta_filepath)
    
    gold_rawlist = [x for xs in gold_texts  for x in xs]
    meta_rawlist = [x for xs in meta_texts  for x in xs]
    gold_df = pd.DataFrame(gold_rawlist, columns =['m_id', 'token', 'token_type', 'gold_status'])
    meta_df = pd.DataFrame(meta_rawlist, columns =['m_id', 'token', 'token_type', 'mention_type', 'gold_status'])
    gold_df_transposed = gold_df.T
    meta_df_transposed = meta_df.T
    
    for i, (j,k) in enumerate(zip(gold_rawlist,meta_rawlist )):

        if len(j)==1:
            continue
        if j[3]!='-' and j[3]!='(0)':

            all_mentions.append(j[3])
            #print(j[3], k[3])
            if j[3].startswith('(') and not j[3].endswith(')'):

                if  gold_rawlist[i+1][3].endswith(')'):
                    two_combo_idx.append([i,i+1])


        #         elif not gold_rawlist[i+1][3].endswith(')'):
                elif gold_rawlist[i+2][3].endswith(')'):
                    #print(j[3], gold_rawlist[i+1][3],gold_rawlist[i+2][3])
                    three_combo_idx.append([i,i+1,i+2])
        #         elif not gold_rawlist[i+2][3].endswith(')'):   
        #             print(gold_rawlist[i+3][3])
                elif gold_rawlist[i+3][3].endswith(')'):

                    #print(j[3], gold_rawlist[i+1][3],gold_rawlist[i+2][3])

                    four_combo_idx.append([i,i+1,i+2, i+3])
                #elif not gold_rawlist[i+3][3].endswith(')'):  #does not apply but just to check 
                elif gold_rawlist[i+4][3].endswith(')'):
                    five_combo_idx.append([i,i+1,i+2, i+3, i+4])
                elif gold_rawlist[i+5][3].endswith(')'):
                    six_combo_idx.append([i,i+1,i+2, i+3, i+4, i+5])
                elif gold_rawlist[i+6][3].endswith(')'):
                    seven_combo_idx.append([i,i+1,i+2, i+3, i+4, i+5, i+6])
                elif gold_rawlist[i+7][3].endswith(')'):
                    #print("yes")
                    eight_combo_idx.append([i,i+1,i+2, i+3, i+4, i+5, i+6, i+7])
                elif gold_rawlist[i+10][3].endswith(')'):
                    #print('yes')
                    elev_combo_idx.append([i,i+1,i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10])     

    #for both gold and meta file, 
    #change the dataframe in indices where bracketted events lie, and make them one row multi-worded event with m_id
    for j in two_combo_idx: 
        
        #print(gold_df_transposed.iloc[1:4,j[0]:j[1]+1 ])
        gold_df_transposed.iloc[1,j[0] ] = gold_df_transposed.iloc[1,j[0] ] + ' ' + gold_df_transposed.iloc[1,j[1] ]  
    for i in three_combo_idx :
         

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]
    for i in four_combo_idx :
        #print(i[0], i[1], i[2])

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]  + ' ' + gold_df_transposed.iloc[1,i[3]]   
    for i in five_combo_idx :
        #print(i[0], i[1], i[2])

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]  + ' ' + gold_df_transposed.iloc[1,i[3]] + ' ' + gold_df_transposed.iloc[1,i[4]]

    for i in six_combo_idx :
        #print(i[0], i[1], i[2])

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]  + ' ' + gold_df_transposed.iloc[1,i[3]] + ' ' + gold_df_transposed.iloc[1,i[4]]+ ' ' + gold_df_transposed.iloc[1,i[5]]

    for i in eight_combo_idx :
        #print(i[0], i[1], i[2])

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]  + ' ' + gold_df_transposed.iloc[1,i[3]] + ' ' + gold_df_transposed.iloc[1,i[4]]+ ' ' + gold_df_transposed.iloc[1,i[5]] + ' ' + gold_df_transposed.iloc[1,i[6]]+ ' ' + gold_df_transposed.iloc[1,i[7]]
    for i in elev_combo_idx :
        #print(i[0], i[1], i[2])

        #print(gold_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        gold_df_transposed.iloc[1,i[0] ] = gold_df_transposed.iloc[1,i[0] ] + ' ' + gold_df_transposed.iloc[1,i[1] ] + ' ' + gold_df_transposed.iloc[1,i[2]]  + ' ' + gold_df_transposed.iloc[1,i[3]] + ' ' + gold_df_transposed.iloc[1,i[4]]+ ' ' + gold_df_transposed.iloc[1,i[5]] + ' ' + gold_df_transposed.iloc[1,i[6]]+ ' ' + gold_df_transposed.iloc[1,i[7]]+ ' ' + gold_df_transposed.iloc[1,i[8]]+ ' ' + gold_df_transposed.iloc[1,i[9]]+ ' ' + gold_df_transposed.iloc[1,i[10]]

    for j in two_combo_idx: 

        meta_df_transposed.iloc[1,j[0] ] = meta_df_transposed.iloc[1,j[0] ] + ' ' + meta_df_transposed.iloc[1,j[1] ]  
    for i in three_combo_idx :


        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]
    for i in four_combo_idx :
        #print(i[0], i[1], i[2])

        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]  + ' ' + meta_df_transposed.iloc[1,i[3]]   
    for i in five_combo_idx :
        #print(i[0], i[1], i[2])

        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]  + ' ' + meta_df_transposed.iloc[1,i[3]] + ' ' + meta_df_transposed.iloc[1,i[4]]

    for i in six_combo_idx :
        #print(i[0], i[1], i[2])

        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]  + ' ' + meta_df_transposed.iloc[1,i[3]] + ' ' + meta_df_transposed.iloc[1,i[4]]+ ' ' + meta_df_transposed.iloc[1,i[5]]

    for i in eight_combo_idx :
        #print(i[0], i[1], i[2])

        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]  + ' ' + meta_df_transposed.iloc[1,i[3]] + ' ' + meta_df_transposed.iloc[1,i[4]]+ ' ' + meta_df_transposed.iloc[1,i[5]] + ' ' + meta_df_transposed.iloc[1,i[6]]+ ' ' + meta_df_transposed.iloc[1,i[7]]
    for i in elev_combo_idx :
        #print(i[0], i[1], i[2])

        #print(meta_df_transposed.iloc[1:4,i[0]:i[2]+1 ])

        meta_df_transposed.iloc[1,i[0] ] = meta_df_transposed.iloc[1,i[0] ] + ' ' + meta_df_transposed.iloc[1,i[1] ] + ' ' + meta_df_transposed.iloc[1,i[2]]  + ' ' + meta_df_transposed.iloc[1,i[3]] + ' ' + meta_df_transposed.iloc[1,i[4]]+ ' ' + meta_df_transposed.iloc[1,i[5]] + ' ' + meta_df_transposed.iloc[1,i[6]]+ ' ' + meta_df_transposed.iloc[1,i[7]]+ ' ' + meta_df_transposed.iloc[1,i[8]]+ ' ' + meta_df_transposed.iloc[1,i[9]]+ ' ' + meta_df_transposed.iloc[1,i[10]]

    #create indices of the rest of the rows to delete them from the gold and meta dataframes, 
    #after multi-worded events are created as one row each
    del_idx_three = [item for sublist in three_combo_idx for item in sublist[1:]]
    del_idx_two = [item for sublist in two_combo_idx for item in sublist[1:]]
    del_idx_four = [item for sublist in four_combo_idx for item in sublist[1:]]
    del_idx_five = [item for sublist in five_combo_idx for item in sublist[1:]]
    del_idx_six = [item for sublist in six_combo_idx for item in sublist[1:]]
    del_idx_eight = [item for sublist in eight_combo_idx for item in sublist[1:]]
    del_idx_elev = [item for sublist in elev_combo_idx for item in sublist[1:]]

    del_list_all = del_idx_three + del_idx_two + del_idx_four + del_idx_five + del_idx_six + del_idx_eight + del_idx_elev
    #print(len(del_list_all))
    
    #delte the extra rows based on the above indices
    gold_df_transposed.drop(gold_df_transposed.iloc[:, del_list_all], axis=1, inplace=True)
    meta_df_transposed.drop(meta_df_transposed.iloc[:, del_list_all], axis=1, inplace=True)
    
    #retranspose the dataframes after deleting those rows as columns
    
    gold_df_retransposed = gold_df_transposed.T
    meta_df_retransposed = meta_df_transposed.T
    #remove the remaining brackets in the gold clusters 
    gold_df_retransposed['gold_status'] = gold_df_retransposed['gold_status'].str.replace('(', '')
    gold_df_retransposed['gold_status'] = gold_df_retransposed['gold_status'].str.replace(')', '')
    meta_df_retransposed['gold_status'] = meta_df_retransposed['gold_status'].str.replace('(', '')
    meta_df_retransposed['gold_status'] = meta_df_retransposed['gold_status'].str.replace(')', '')
    
    gold_df_retransposed = gold_df_retransposed.reset_index()
    meta_df_retransposed = meta_df_retransposed.reset_index()
    
    
    #create csv file which contains both gold and meta data
    mention_csv_all = meta_df_retransposed.loc[(meta_df_retransposed['gold_status'] != '-') & (~meta_df_retransposed['gold_status'].isnull())]
    
    # get list of indices where 'NEWLINE' has to be added as new rows
    marked = gold_df_retransposed.loc[gold_df_retransposed['m_id'].str.startswith('#e')].index.tolist()

    # create insert template row, using a random m_id for the added new NEWLINE rows just as a token. 
    insert_gold = pd.DataFrame({'m_id':['40b69cf630792394ef847aee6c959ece.t1.1'],'token':'NEWLINE','token_type':'BODY', 'gold_status':['-']})
    insert_meta = pd.DataFrame({'m_id':['40b69cf630792394ef847aee6c959ece.t1.1'],'token':'NEWLINE','token_type':'BODY', 'mention_type':[np.nan],'gold_status':['-']})
     
    # loop through marked indices and insert row
    for x in marked:
        gold_df_retransposed = pd.concat([gold_df_retransposed.loc[:x-1],insert_gold ,gold_df_retransposed.loc[x:]])
        meta_df_retransposed = pd.concat([meta_df_retransposed.loc[:x-1],insert_meta,meta_df_retransposed.loc[x:]])
    # finally reset the index and spit out new dataframe
    gold_df_retransposed = gold_df_retransposed.reset_index(drop=True)
    meta_df_retransposed = meta_df_retransposed.reset_index(drop=True)
    #print("final mentions list",gold_df_retransposed.loc[(gold_df_retransposed['gold_status'] != '-') & (~gold_df_retransposed['gold_status'].isnull())])
    #drop index column and then convert dataframe to list for further preprocessing i.e to create the maps
    gold_df_retransposed = gold_df_retransposed.iloc[: , 1:]
    
    meta_df_retransposed = meta_df_retransposed.iloc[: , 1:]
    gold_df_retransposed_list = gold_df_retransposed.values.tolist()
    meta_df_retransposed_list = meta_df_retransposed.values.tolist()
    print("final lists created")
    return gold_df_retransposed_list, meta_df_retransposed_list, mention_csv_all
 

    
def create_doc_sent_map(gold_filepath, meta_filepath):
    
    """
    Create the document to sentence mapping for GVC corpus'

    Parameters
    ----------
    gold_filepath: Path to the GVC Gold Annotation Directory
    meta_filepath : Path to the GVC Verbose(meta data for event type) Annotation Directory
 
    Returns
    -------
    doc_sent_map : dict
    """
    
    #create the processed gold and meta files to combine bracketted mentions 
    gold_rawlist,meta_rawlist, mention_csv_all = create_gold_metaraw(gold_filepath, meta_filepath)
    print("Gold and Raw files processed")

    doc_id  = []
    mention_id = []
    start_index = []
    end_index = []
    doc_map =defaultdict(dict) 
     
    doc_sent_map =defaultdict(dict) 

    sent_dict = {}
    token_dict = defaultdict(dict) 
    men_dict = defaultdict(dict) 
    mentions_list = []
    list_keys = []
    dummy_m_id = []

    # manually found the doc_IDS for which sentences aren't separted by NEWLINE, since we are separating bert sentences 
    #by 'NEWLINE' token. This inconsistency is due to how raw GVC files were released in https://github.com/cltl/GunViolenceCorpus
    NO_NEWLINE_docIDS = ['(c7e44056252cdb25cb66f270a3cc8f8d);',
 '(00f9d7565109c261b4621f09d5934d6f);',
 '(eb18c320af527c52fd840998ed608d93);',
 '(280357b24aa337bad393bec8d53c9b03);',
 '(be7dcfcc71880be1e3e8c5be239b3ea4);',
 '(662672852b984c058a02869c92096d86);',
 '(9c0d088f9268bd3aca1d017b705835f5);',
 '(51dfbd24c13b17a0c7ca70a839c7e94f);',
 '(5c24e31c9271a107c04e3618d926a457);',
 '(471ea29c5998fd1deb08fbaddb292bcc);',
 '(ecc398cc16f22eb1323b4411ef866651);',
 '(1424b8bc9e983d29ce655860d1d3d380);',
 '(af2169d4613cbdb7219b3c0f177526b2);']       
            
    #Loop through the gold and raw files to extract the doc_sent_map and mention_map in the earlier format for 
    #Ecbplus and LDC
            
    for i, j in enumerate(gold_rawlist):
        if "#b" in j[0]:
            start_index.append(i)
            doc_map[str(j[0]).split(' ')[2] ]['start_index'] = i

        if "#e" in j[0]:
            end_index.append(i)
         
        elif not "#b" in j[0] and not "#e" in j[0] and j[3]!='-':
            mention_id.append(j[3])
 
    for j,k in  enumerate(doc_map.values()):
        
        k['end_index'] = end_index[j] 
    l=[]   
    doc_count=0
    doc_chunk = []
    bert_sent = []
    text_start_char = []
    token_len = 0
    pattern = "[\d+[0-9]{1,2}.[0-9]{1,2}$"
    for j,k in  enumerate(doc_map.items()):
       
        doc_chunk = gold_rawlist[k[1]['start_index'] +1:k[1]['end_index']]  #gold data

        meta_chunk = meta_rawlist[k[1]['start_index'] +1:k[1]['end_index']]  #verbose or meta data
        #print(meta_chunk)
        count_mention = 0
        for x,y in zip(doc_chunk,meta_chunk ) : #replace these

            if 'b' or 't' in x[0][-6:]:
               
                match = re.search(pattern, x[0][-6:])   
                if match is not None:
                     
                    line_break = match.group(0).split('.')[0] 
                    
                    sent_dict[token_len] = x[1]
                    token_dict [x[0]]['token_start_char']  = token_len
                    dummy_m_id.append(x[0])
                    
                    token_len+=len(x[1])+1
                    start_char = [w for i, w in enumerate(sent_dict.keys()) if i ==0]
                    text_start_char.append([w for i, w in enumerate(sent_dict.keys()) if x[3] !='-'])
                    if x[3]!='-':

                        mentions_list.append([x[0], k[0],x[1],x[3], y[3], start_char[0] ] )
        #FOR FIRST DOC with PERIOD

                    if doc_count==0 and x[1] =='.':
                         
                        doc_sent_map[k[0]][start_char[0]] = {

            'doc_id': k[0],
                'sent_text': ' '.join([x for x in sent_dict.values()]),
                 'token_map':{t1:t2 for t1,t2 in sent_dict.items()},
                'start_char': start_char[0],
                'sent_id': match.group(0).split('.')[0],
                'token_id': {t1:t2 for t1,t2 in token_dict.items()}}
                        sent_dict.clear()
                        token_dict.clear()
                      
    #FOR REST Documents with NEWLINE
                    elif doc_count>=1 and x[1] =='NEWLINE'  :
                     
                        doc_sent_map[k[0]][start_char[0] ] = {

            'doc_id': k[0],
                'sent_text': ' '.join([x for x in sent_dict.values()]),
                 'token_map':{t1:t2 for t1,t2 in sent_dict.items()},
                'start_char': start_char[0],
                'sent_id': match.group(0).split('.')[0],
                            'token_id': {t1:t2 for t1,t2 in token_dict.items()} }
                        sent_dict.clear()
                        token_dict.clear()
                    elif k[0] in NO_NEWLINE_docIDS: # found manually before, this is the list of documents where sentences are not separated by NEWLINE but '.'
                        if x[1] =='.':
                            doc_sent_map[k[0]][start_char[0] ] = {

            'doc_id': k[0],
                'sent_text': ' '.join([x for x in sent_dict.values()]),
                 'token_map':{t1:t2 for t1,t2 in sent_dict.items()},
                'start_char': start_char[0],
                'sent_id': match.group(0).split('.')[0],
                                'token_id': {t1:t2 for t1,t2 in token_dict.items()}}
                            sent_dict.clear()
                            token_dict.clear()
            else:
                    print('no match')
                    print(doc_count)
        token_len =0
        sent_dict.clear()
        doc_count+=1
        
    return doc_sent_map,mentions_list
 

def add_bert_docs(mention_map, doc_sent_map):
    """
    Add the entire document of the mention with the sentence corresponding to the mention
    replaced by the bert_sentence (sentence in which the mention is surrounded by <m> and </m>)
    The key for this is 'bert_doc'

    Parameters
    ----------
    mention_map: dict
    doc_sent_map : dict

    Returns
    -------
    None
    """
    # for each mention create and add bert_doc
    for mention in mention_map.values():
        m_sentence_start_char = mention['sentence_start_char']
        doc_id = mention['doc_id']
        # create the copy of doc_id's sent_map
        m_sent_map = copy.deepcopy(doc_sent_map[doc_id])
        # replace the sentence associated with the mention with bert_sentence
        m_sent_map[m_sentence_start_char]['sent_text'] = mention['bert_sentence']
        # convert sent_map to text
        bert_doc = '\n'.join([sent_map['sent_text'] for sent_map in m_sent_map.values()])
        # add bert_doc in mention
        mention['bert_doc'] = bert_doc.replace("NEWLINE\n", "").replace(" NEWLINE", "")


def get_mention_map_from_mentionlist(mentions_list, doc_sent_map):
    """

    Parameters
    ----------
    mentions_list: list
        List of mentions along with their meta data
 
    doc_sent_map: dict
        The dictionary containing the mapping of sentences in a document
  
    Returns
    -------
    dict: The dictionary of the mentions and document map
    """
    mention_start = [] #start indices of event mentions 
    mention_csv = pd.DataFrame(mentions_list,  columns =['m_id', 'doc_id', 'mention_text','gold_cluster', 'event_type',

                                                    'sentence_start_char'])
    mention_list = list(mention_csv['m_id'])
    
    for i in doc_sent_map.values():
        for j in i.values():

            for k,m in list(j['token_id'].items()):

                for l in mention_list:

                    if k == l:

                        mention_start.append(m['token_start_char'])
                        
                        
    #get end character for the mention using the start character index and adding the length of the mention text 
    mention_csv['textoffset_startchar'] = mention_start
    mention_csv['textoffset_endchar'] = mention_csv.apply(lambda x: x['textoffset_startchar'] + len(x['mention_text']), axis=1)
    #replace remaining brackets with empty space if needed
    mention_csv['gold_cluster'] = mention_csv.apply(lambda x: x['gold_cluster'].replace("(", ""), axis=1)
    mention_csv['gold_cluster'] = mention_csv.apply(lambda x: x['gold_cluster'].replace(")", ""), axis=1)
    
    mention_ids = set([row for row in mention_csv['m_id']])
    mention_to_m_id = {id_: i for i, id_ in enumerate(mention_ids)}
    #convert csv into rows for creating the initial mention map 
    mention_csv_rows = mention_csv.values.tolist()
    mention_map = {
            row[0]: {
                'm_id': mention_to_m_id[row[0]],
                'mention_id': row[0],
                'doc_id': row[1],
                'men_type': row[4],
                'start_char': row[6],
                'end_char': row[7],
                'gold_cluster': row[3],
                'mention_text': row[2],
                'topic': None,
                'lang_id': 'English',
                'sentence_start_char': int(row[5]),
            } for row in mention_csv_rows
        }

      # add <m> and </m> around the trigger text of the mention
    for m in mention_map.values():
        if m['doc_id'] in doc_sent_map:
            m_start_char = m['start_char']
            m_end_char = m['end_char'] + 1
            #sent_map = get_sentence(m['doc_id'], m_start_char, doc_sent_map)
            #print("sent map",sent_map)
            s_start_char = m['sentence_start_char']
           
            sent_map = doc_sent_map[m['doc_id']][s_start_char] 
            sent_text = str(sent_map['sent_text'])
            m['mention_text'] = sent_text[m_start_char - s_start_char: m_end_char - s_start_char]
            
            m['sentence'] = sent_text.replace(" NEWLINE", "")
            m['sent_id'] = sent_map['sent_id']
            m['bert_sentence'] = sent_text[: m_start_char - s_start_char] + '<m> ' +                                  m['mention_text'] + '</m> ' +                                  sent_text[m_end_char - s_start_char:].replace(" NEWLINE", "")
     
    add_bert_docs(mention_map, doc_sent_map)

    return mention_map, doc_sent_map   

def extract_mentions(gold_dir, meta_dir, working_folder):
    """
    Extracts the mention information from the ltf files using the annotations tab file

    Parameters
    ----------
    gold_dir : str
        Path to the GVC Gold Annotation Directory

    meta_dir: str
        Path to the GVC Verbose(meta data for event type) Annotation Directory
        
    working_folder: str
        Path to the working folder to save intermediate files

    Returns
    -------
    dict

        dict of dicts representing the mentions
        dict has these keys: {mention_id, mention_text, doc_id, sent_id, sent_tok_numbers, doc_tok_numbers}
    """
    # generate mention maps
    eve_mention_map_file = working_folder + '/gvc_mention_map.pkl'
    doc_sent_map_file = working_folder + '/gvc_doc_sent_map.pkl'
    
    if os.path.exists(eve_mention_map_file) and os.path.exists(doc_sent_map_file):
        # if files already there, just load the pickles
        gvc_mention_map = pickle.load(open(eve_mention_map_file, 'rb'))
        doc_sent_map = pickle.load(open(doc_sent_map_file , 'rb'))
    else:
        # read the GVC annotated gold and meta data files
        
        #gold_rawlist,meta_rawlist,mentions_all = create_gold_metaraw(gold_filepath, meta_filepath)
        doc_sent_map, mentions_list = create_doc_sent_map(gold_filepath, meta_filepath)
        gvc_mention_map, doc_sent_map  = get_mention_map_from_mentionlist(mentions_list, doc_sent_map)

        # pickle them
        pickle.dump(gvc_mention_map, open(eve_mention_map_file, 'wb'))
        pickle.dump(doc_sent_map, open(doc_sent_map_file, 'wb'))
    
    return gvc_mention_map, doc_sent_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the GVC Gold and Meta files to     generate a common representation of events')
    parser.add_argument('--gold', '-a', help='Path to the GVC Gold Annotation Directory')
    parser.add_argument('--meta', '-s', help='Path to the GVC Verbose(meta data for event type) Annotation Directory')
    args = parser.parse_args()
    print("Using the Gold Directory: ", args.gold)
    print("Using the Source     Directory: ", args.meta)
    gold_filepath = "/s/chopin/d/proj/ramfis-aida/gvc/GunViolenceCorpus/gold.conll"
    meta_filepath = "/s/chopin/d/proj/ramfis-aida/gvc/GunViolenceCorpus/verbose.conll"
    working_folder = '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/tmp_folder'
    #gvc_mention_map, doc_sent_map = extract_mentions(args.ann, args.source, working_folder)  
    gvc_mention_map, doc_sent_map = extract_mentions(gold_filepath, meta_filepath, working_folder)      
   






