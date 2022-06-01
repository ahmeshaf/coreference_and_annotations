#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Author: Rehan
import pickle
#from preprocess import extract_mentions_aida
import os.path
import pyhocon
#from models import *
import torch
from tqdm import tqdm
from cdlm_bert_models.cross_encoder import FullCrossEncoder, FullCrossEncoderSingle
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *

import configparser

config_file_path = '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/model_parameters.ini'
config_model = configparser.ConfigParser()
config_model.read('/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/model_parameters.ini')

#roberta_dir = './models/roberta-experiment/roberta-base-4096/'


def generate_cdlm_embeddings_from_model(parallel_model,bert_model , working_folder,men_type, bert_sentences, device, batch_size=150):
    """

    Parameters
    ----------
    parallel_model: Parallel Model
    bert_sentences: arr
        list of sentences with mentions surrounded by <m> and </m>
    device: torch.device
    batch_size: int

    Returns
    -------
        arr of vectors
    """
    
    vector_map_path = working_folder + f'/{men_type}_{bert_model}_vectormaps.pkl'
    
    if os.path.exists(vector_map_path) :
        # if files already there, just load the pickles
        all_vectors = pickle.load(open(vector_map_path, 'rb'))
        
        
        
    else:
        parallel_model.eval()

        # Store vectors here
        all_vectors = []

        # find the locations where the event/ent mentions exceed max token length of 512 

        loc_mentions = []

        batch = [parallel_model.module.tokenizer.cls_token + bert_sentences[x] + parallel_model.module.tokenizer.sep_token for x in range( len(bert_sentences))]

        bert_tokens = parallel_model.module.tokenizer(batch, pad_to_max_length=True, add_special_tokens=False, truncation =False )
        #bert_tokens = parallel_model.module.tokenizer(batch, padding = 'max_length', add_special_tokens=False, truncation =True )
        input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
        attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
        m = input_ids.cpu()
        k = m == parallel_model.module.vals[0]
        p = m == parallel_model.module.vals[1]
        v = (k.int() + p.int()).bool()
        nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)
        #print(nz_indexes.shape)
        #print(nz_indexes[0][1])
        loc_mentions.extend(nz_indexes.cpu().numpy())
        #print(input_ids.shape)

        # get indices of overflowing tokens 



        if men_type == 'evt':
            overflow_idx=[]
            #[x[1] for x in loc_mentions]
            for i,j in enumerate(loc_mentions):
                if j[1]>=512:
                    overflow_idx.append([i, j[1] ])
            #print(overflow_idx)



            idx = [x[0] for x in overflow_idx]
            m_new  = []
            for i, j in enumerate(overflow_idx):
                bert_sent = bert_sentences[j[0] ]
                m_new.append(m[j[0],  j[1]-512+1:j[1]+1] )

            m_new_tensor = torch.stack(m_new) 
            m_new_tensor = F.pad(m_new_tensor,pad =(0, m.shape[1]-m_new_tensor.shape[1]), value=1) 

            m[idx] = m_new_tensor
            v = (k.int() + p.int()).bool()
            nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)

            q = torch.arange(m.shape[1])
            q = q.repeat(m.shape[0], 1)

            msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
            msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q

            msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
            msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q

            attention_mask_g = msk_0.int() * msk_1.int()

            input_ids = input_ids[:, :512]
            # attention_mask = attention_mask[:, :4096]
            #attention_mask[:, 0] = 2
            #attention_mask[attention_mask_g == 1] = 2
            attention_mask = attention_mask[:, :512]
            arg1 = msk_0_ar.int() * msk_1_ar.int()
            arg_new1 = msk_0.int() * msk_1.int()
            arg_new1 = arg_new1[:,:512]
            arg1 = arg1[:, :512]
            arg1 = arg1.to(device)


            with torch.no_grad():
                for i in tqdm(range(0, len(bert_sentences), batch_size), desc="generating "):

                    men_vectors = parallel_model(input_ids[i:i+batch_size], attention_mask[i:i+batch_size], arg1[i:i+batch_size]) 
                    all_vectors.extend(men_vectors)



        elif men_type == 'ent':

            batch = [parallel_model.module.tokenizer.cls_token + bert_sentences[x] + parallel_model.module.tokenizer.sep_token for x in range( len(bert_sentences))]

            bert_tokens = parallel_model.module.tokenizer(batch, pad_to_max_length=True, add_special_tokens=False, truncation =False )
            #bert_tokens = parallel_model.module.tokenizer(batch, padding = 'max_length', add_special_tokens=False, truncation =True )
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
            m = input_ids.cpu()
            k = m == parallel_model.module.vals[0]
            p = m == parallel_model.module.vals[1]
            v = (k.int() + p.int()).bool()
            nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)
            q = torch.arange(m.shape[1])
            q = q.repeat(m.shape[0], 1)

            msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
            msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q

            msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
            msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q

            attention_mask_g = msk_0.int() * msk_1.int()

            input_ids = input_ids[:, :512]
            # attention_mask = attention_mask[:, :4096]
            #attention_mask[:, 0] = 2
            #attention_mask[attention_mask_g == 1] = 2
            attention_mask = attention_mask[:, :512]
            arg1 = msk_0_ar.int() * msk_1_ar.int()
            arg_new1 = msk_0.int() * msk_1.int()
            arg_new1 = arg_new1[:,:512]
            arg1 = arg1[:, :512]
            arg1 = arg1.to(device)

            with torch.no_grad():
                for i in tqdm(range(0, len(bert_sentences), batch_size), desc="generating "):

                    men_vectors = parallel_model(input_ids[i:i+batch_size], attention_mask[i:i+batch_size], arg1[i:i+batch_size]) 
                    all_vectors.extend(men_vectors)
        




#       

    
#     if not os.path.exists(working_folder):
#         os.makedirs(working_folder)

#     # generate the eve_mention_map
#     eve_mention_map_path = working_folder + '/eve_mention_map.pkl'
#     if not os.path.exists(eve_mention_map_path):
#         eve_mention_map = extract_mentions_aida(mentions_csv_path, source_dir, output_folder)
#         pickle.dump(eve_mention_map, open(eve_mention_map_path, 'wb'))
#     else:
#         eve_mention_map = pickle.load(open(eve_mention_map_path, 'rb'))

    
    #create vector map path and save them
   
        
   
        # save them if they are not already pickled
        
    
        pickle.dump(all_vectors, open(vector_map_path, 'wb'))
        

        

    
    
     
    #vector_map = {m_id: vec for m_id, vec in zip(men_ids, vectors)}
    #pickle.dump(all_vectors, open(vector_map_path, 'wb'))

    return all_vectors


def generate_cdlm_embeddings(config_file_path,working_folder, bert_model,bert_sentences,men_type, num_gpus=4, batch_size=150, cpu=False, *args, **kwargs):
    """
    Generate word embeddings for various bert models 

    Parameters
    ----------
    config_file_path: str
        Path to the custom bert config files

    bert_model: str
        Path to the bert model weights

    bert_sentences: arr
        list of sentences with mentions surrounded by <m> and </m>

    num_gpus: int
        Number of gpus to use

    batch_size:
        Batch size

    cpu: bool
        If we want to use cpu instead of gpus

    Returns
    -------
    tensors
        list of tensors, ie, the representations of the mentions
    """
    #config = pyhocon.ConfigFactory.parse_file(config_file_path)
    device_ids = [0,1,2,3]
    #model_coref = AutoModel.from_pretrained('nielsr/coref-bert-base')

    device = torch.device("cuda:{}".format(device_ids[0]))
    if cpu:
        device = torch.device("cpu")
    cross_encoder = FullCrossEncoderSingle(config_model , long=True, coref_model=bert_model)
    cross_encoder = cross_encoder.to(device)
    #cross_encoder.model = AutoModel.from_pretrained(os.path.join(bert_dir, 'bert')).to(device)
    #cross_encoder.model = AutoModel.from_pretrained(os.path.join(roberta_dir)).to(device)
     
    parallel_model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
    parallel_model.module.to(device)
    parallel_model.module.tokenizer
    #parallel_model = torch.nn.parallel.DistributedDataParallel(cross_encoder, device_ids=device_ids)
    #parallel_model = torch.nn.parallel.DistributedDataParallel(cross_encoder, device_ids=None)
    
    if cpu:
        parallel_model.module.to(device)
        
    
    #return parallel_model
    return generate_cdlm_embeddings_from_model(parallel_model,bert_model,working_folder,men_type,  bert_sentences, device, batch_size)


def generate_cdlm_embeddings_aida(mentions_csv_path, source_dir, config_file_path, bert_model_path, output_folder):
    """
    generate the embeddings for the aida mentions extracted from

    Parameters
    ----------
    mentions_csv_path : str
        The path of the csv file containing offset information for mentions.
        This is generated by the aida generate-triples script

    source_dir: str
        The source directory of the LDC source (ltf and topics files)

    config_file_path: str
        The path for the LongFormer/BERT config file

    output_folder: str
        Path to the output folder

    bert_model_path: str
        Path to the bert model weights

    Returns
    -------
    None. Save embeddings in the output folder
    """
    # create temp working folder output folder
    working_folder = output_folder + '/WORKING/'
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    # generate the eve_mention_map
    eve_mention_map_path = working_folder + '/eve_mention_map.pkl'
    if not os.path.exists(eve_mention_map_path):
        eve_mention_map = extract_mentions_aida(mentions_csv_path, source_dir, output_folder)
        pickle.dump(eve_mention_map, open(eve_mention_map_path, 'wb'))
    else:
        eve_mention_map = pickle.load(open(eve_mention_map_path, 'rb'))

    vector_map_path = working_folder + '/cdlm_vector_map.pkl'
    men_ids, bert_sentences = zip(*[(men_id, men['bert_sentence']) for men_id, men in eve_mention_map.items()])
    vectors = generate_cdlm_embeddings(config_file_path, bert_model_path, list(bert_sentences))
    vector_map = {m_id: vec for m_id, vec in zip(men_ids, vectors)}
    pickle.dump(vector_map, open(vector_map_path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the LDC2019E77 Annotations and LDC2019E42 source files to \
    generate a common representation of events')
    parser.add_argument('--ann', '-a', help='Path to the LDC Annotation Directory')
    parser.add_argument('--source', '-s', help='Path to the LDC Source Directory')
    parser.add_argument('--tmp_folder', '-t', default='./tmp', help='Path to a working directory')
    args = parser.parse_args()
    print("Using the Annotation Directory: ", args.ann)
    print("Using the Source     Directory: ", args.source)

    #extract_mentions(args.ann, args.source, './')
    generate_cdlm_embeddings(config_file_path,args.tmp_folder, bert_model,bert_sentences, num_gpus=4, batch_size=150, cpu=False, *args, **kwargs)
