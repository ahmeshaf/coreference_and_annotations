 

import os
import sys
import gc
import numpy as np
import csv
gc.collect()
import torch
import torch.nn as nn
import torch.nn as F
from collections import defaultdict
#torch.cuda.empty_cache()
 
print(torch.cuda.current_device())
#parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')

parent_path = '/s/chopin/d/proj/ramfis-aida//coref/coreference_and_annotations/'

sys.path.append(parent_path)

import os.path
import pickle

from sklearn.model_selection import train_test_split
import pyhocon
from coreference.triplet_model import LongFormerCrossEncoder, LongFormerCosAlign,LongFormerTriplet
 
import random
from tqdm.autonotebook import tqdm
from parsing.parse_ecb import parse_annotations
 

def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(predicted_labels)


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(true_labels)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    return 2 * P * R / (P + R)


def load_data(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            triviality_label = 0 if row[2] =='HARD' else 1
            
            all_examples.append((mention_pair, triviality_label))

    return all_examples

#load lemma balanced TP and FP tsv pairs

def load_data_tp_fp(trivial_non_trivial_path):
    all_examples = []
    pos = []
    neg = []
    with open(trivial_non_trivial_path) as tnf:
        rd = csv.reader(tnf, delimiter="\t", quotechar='"')
        
        for line in rd:
            #row = line.strip().split(',')
            mention_pair = line[:2]
            #print(line[2])
            #print(mention_pair)
            if line[2] =='POS':
                triviality_label = 1
                all_examples.append((mention_pair, triviality_label))
                #pos.append(mention_pair)
                
            else:
                triviality_label = 0
                all_examples.append((mention_pair, triviality_label))
                #neg.append(mention_pair)
          
    return all_examples 

def load_data_cross_full(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            triviality_label = 0 if row[3] =='NEG' else 1

            all_examples.append((mention_pair, triviality_label))

    return all_examples

def load_data_pair_coref_dev(trivial_non_trivial_path):
    all_examples = []
    #condition to select only hard pos and hard neg examples 
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            if row[2]=='HARD':
                #triviality_label = int(row[3])
                triviality_label = 0 if row[3] =='NEG' else 1

                all_examples.append((mention_pair, triviality_label))

    return all_examples
def load_data_pair_coref(trivial_non_trivial_path):
    all_examples = []
    #condition to select only hard pos and hard neg examples 
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            
            if row[2]=='HARD':
                
            #triviality_label = int(row[3])
                triviality_label = -1 if row[3] =='NEG' else 1

                all_examples.append((mention_pair, triviality_label))

    return all_examples


def print_label_distri(labels):
    label_count = {}
    for label in labels:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    print(len(labels))
    label_count_ratio = {label: val / len(labels) for label, val in label_count.items()}
    return label_count_ratio


def split_data(all_examples, dev_ratio=0.2):
    pairs, labels = zip(*all_examples)
    return train_test_split(pairs, labels, test_size=dev_ratio)


def tokenize(tokenizer, mention_pairs, mention_map,m_start, m_end, max_sentence_len=512, context = "bert_doc"):
 
    if max_sentence_len is None:
        
        max_sentence_len = tokenizer.model_max_length #try 512 here, 
        
    #max_sentence_len=2048 #trying out a greater context since Longformer! 

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
        
 
        if context =="bert_doc":

            sentence_a = mention_map[m1]['bert_doc'].replace("\n", "")
            sentence_b = mention_map[m2]['bert_doc'].replace("\n", "")
        else:
            sentence_a = mention_map[m1]['bert_sentence'] 
            sentence_b = mention_map[m2]['bert_sentence'] 

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]),                    ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        
        if context == "bert_sentence":
            
            for input_id in input_ids:
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
                in_truncated = input_id[m_end_index-(max_sentence_len//4): m_end_index] +                                input_id[m_end_index: m_end_index + (max_sentence_len//4)]


                in_truncated = in_truncated + [tokenizer.pad_token_id]*(max_sentence_len//2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)
        else:
            
            #print(context)
            for input_id in input_ids:
            
                global_input_id = [1]
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
                doc_start_token = [50267]
                doc_end_token =  [50268]

               
                in_truncated = input_id[m_start_index-(max_sentence_len//4): m_start_index] +                 input_id[m_start_index:m_end_index+1] + input_id[ m_end_index+1: m_end_index + (max_sentence_len//4)]
                
#                 in_truncated = input_id[0: m_start_index] + \
#                 input_id[m_start_index:m_end_index+1] + input_id[ m_end_index+1: m_end_index + (max_sentence_len//4-2)]

                in_truncated = global_input_id + doc_start_token + in_truncated + doc_end_token
#                 new_ids = torch.cat((doc_start_token , in_truncated), dim=1)
#                 new_ids = torch.cat((global_input_id, new_ids), dim=1)
#                 new_ids = torch.cat((new_ids, doc_end_token), dim=1)
                #in_truncated = new_ids 
# #                  
# #                               

                in_truncated = in_truncated + [tokenizer.pad_token_id]*((max_sentence_len//2) - len(in_truncated))
                #print(len(in_truncated[0:256]))
                #input_ids_truncated.append(in_truncated[0:1024])  
                input_ids_truncated.append(in_truncated[0:1024])   

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
    tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)

    return tokenized_ab, tokenized_ba

def tokenize_triplets(tokenizer, mention_triplets, mention_map,m_start, m_end, max_sentence_len=512, context = "bert_doc"):
 
    if max_sentence_len is None:
        
        max_sentence_len = tokenizer.model_max_length #try 512 here, 
        
    #max_sentence_len=2048 #trying out a greater context since Longformer! 

    pairwise_bert_instances_aa = []
    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ac = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2, m3) in mention_triplets:
        
 
        if context =="bert_doc":

            sentence_a = mention_map[m1]['bert_doc'].replace("\n", "")
            sentence_b = mention_map[m2]['bert_doc'].replace("\n", "")
            sentence_c = mention_map[m3]['bert_doc'].replace("\n", "")
        else:
            #print(m1, m2, m3)
            sentence_a = mention_map[m1]['bert_sentence'] 
            #print("sentence A", sentence_a )
            sentence_b = mention_map[m2]['bert_sentence'] 
            #print("sentence B", sentence_b )
            sentence_c = mention_map[m3]['bert_sentence'] 
            #print("sentence C", sentence_c )
            

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]),                    ' '.join([doc_start, sent_b, doc_end])

        
        instance_aa = make_instance(sentence_a, sentence_a)
        #print("sentence aa", instance_aa)
        pairwise_bert_instances_aa.append(instance_aa)
        
        instance_ab = make_instance(sentence_a, sentence_b)
        #print("sentence ab",instance_ab)
        pairwise_bert_instances_ab.append(instance_ab)
        
        instance_ac = make_instance(sentence_a, sentence_c)
        #print("sentence ac", instance_ac)
        pairwise_bert_instances_ac.append(instance_ac)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        
        if context == "bert_sentence":
            
            for input_id in input_ids:
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
                in_truncated = input_id[m_end_index-(max_sentence_len//4): m_end_index] +                                input_id[m_end_index: m_end_index + (max_sentence_len//4)]


                in_truncated = in_truncated + [tokenizer.pad_token_id]*(max_sentence_len//2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)
        else:
            
   
            for input_id in input_ids:
           
                global_input_id = [1]
                m_start_index = input_id.index(m_start)
                m_end_index = input_id.index(m_end)
   
                doc_start_token = [50267]
                doc_end_token =  [50268]

                in_truncated = input_id[m_start_index-(max_sentence_len//4): m_start_index] +                 input_id[m_start_index:m_end_index+1] + input_id[ m_end_index+1: m_end_index + (max_sentence_len//4)]
                
#                 in_truncated = input_id[0: m_start_index] + \
#                 input_id[m_start_index:m_end_index+1] + input_id[ m_end_index+1: m_end_index + (max_sentence_len//4-2)]

                in_truncated = global_input_id + doc_start_token + in_truncated + doc_end_token
# 

                in_truncated = in_truncated + [tokenizer.pad_token_id]*((max_sentence_len//2) - len(in_truncated))
            
                #input_ids_truncated.append(in_truncated[0:1024])  
                input_ids_truncated.append(in_truncated[0:256])   

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b,  = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)
        #tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    tokenized_aa = ab_tokenized(pairwise_bert_instances_aa)
    tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
    tokenized_ac = ab_tokenized(pairwise_bert_instances_ac)
  
    return tokenized_aa, tokenized_ab, tokenized_ac


def get_arg_attention_mask(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    input_ids.cpu()

    num_inputs = input_ids.shape[0]
    m = input_ids.cpu()

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id
    
    k = m == parallel_model.module.vals[0]
    p = m == parallel_model.module.vals[1]
    v = (k.int() + p.int()).bool()
 
    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)
    

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    
    # I think CLS token should also have global attention apart from the mentions 
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()
    
    
 

    return attention_mask_g, arg1, arg2


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                               global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only)


def generate_lm_out(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    ab_lm_out_all = []
    ba_lm_out_all = []
    new_batch_size = batching(n, batch_size, len(device_ids))
    batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Generating LM Outputs"):
            batch_indices = indices[i: i + batch_size]
            lm_out_ab = forward_ab(parallel_model, dev_ab, device, batch_indices, lm_only=True).detach().cpu()
            ab_lm_out_all.append(lm_out_ab)

            lm_out_ba = forward_ab(parallel_model, dev_ba, device, batch_indices, lm_only=True).detach().cpu()
            ba_lm_out_all.append(lm_out_ba)

    return {'ab': torch.vstack(ab_lm_out_all), 'ba': torch.vstack(ba_lm_out_all)}


def frozen_predict(parallel_model, device, dev_ab, dev_ba, batch_size, lm_output_file_path, force_lm_output=False):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    if not os.path.exists(lm_output_file_path) or force_lm_output:
        lm_out_dict = generate_lm_out(parallel_model, device, dev_ab, dev_ba, batch_size)
        pickle.dump(lm_out_dict, open(lm_output_file_path, 'wb'))
    else:
        lm_out_dict = pickle.load(open(lm_output_file_path, 'rb'))

    new_batch_size = batching(n, batch_size, len(device_ids))
    batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Predicting"):
            batch_indices = indices[i: i + batch_size]
            ab_out = lm_out_dict['ab'][batch_indices, :]
            ba_out = lm_out_dict['ba'][batch_indices, :]
            ab_out.to(device)
            ba_out.to(device)
            scores_ab = parallel_model(ab_out, pre_lm_out=True)
            scores_ba = parallel_model(ba_out, pre_lm_out=True)
            scores_mean = (scores_ab + scores_ba)/2
            batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)


def cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    cos_dev = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
     

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            cos_ab = forward_ab(parallel_model, dev_ab, device, batch_indices) # these scores are actually embeddings 
            cos_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            batch_predictions = cos_dev(cos_ab, cos_ba).detach().cpu()
            #print("batch dev prediction tensor", batch_predictions)
            batch_predictions = (batch_predictions > 0.48).detach().cpu()
       
            predictions.append(batch_predictions)

    return torch.cat(predictions)


def predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices) 
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)

            scores_mean = (scores_ab + scores_ba) / 2

            batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)

def predict_cross_scores(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions_ab = []
    predictions_ba = []
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices).detach().cpu() 
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices).detach().cpu()

            #scores_mean = (scores_ab + scores_ba) / 2

            #batch_predictions = (scores_mean > 0.5).detach().cpu()
            batch_predictions_ab = scores_ab.detach().cpu()
            batch_predictions_ba = scores_ba.detach().cpu()
            predictions_ab.append(batch_predictions_ab)
            predictions_ba.append(batch_predictions_ba)
    return torch.cat(predictions_ab), torch.cat(predictions_ba)
    #return predictions
def train_frozen(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          working_folder,
          device,
          force_lm_output=False,
          batch_size=30,
          n_iters=10,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])
    tokenizer = parallel_model.module.tokenizer
    # prepare data
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id)
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)

    lm_output_file_path_train = working_folder + '/lm_output_train.pkl'
    lm_output_file_path_dev = working_folder + '/lm_output_dev.pkl'

    if not os.path.exists(lm_output_file_path_train) or force_lm_output:
        lm_out_dict = generate_lm_out(parallel_model, device, train_ab, train_ba, batch_size)
        pickle.dump(lm_out_dict, open(lm_output_file_path_train, 'wb'))
    else:
        lm_out_dict = pickle.load(open(lm_output_file_path_train, 'rb'))

    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]
            ab_out = lm_out_dict['ab'][batch_indices, :]
            ba_out = lm_out_dict['ba'][batch_indices, :]
            scores_ab = parallel_model(ab_out.to(device), pre_lm_out=True)
            scores_ba = parallel_model(ba_out.to(device), pre_lm_out=True)
            scores_mean = (scores_ab + scores_ba) / 2
            batch_labels = train_labels[batch_indices].to(device)
            loss = bce_loss(torch.squeeze(scores_mean), batch_labels) + mse_loss(scores_ab, scores_ba)
            loss.backward()
            optimizer.step()
            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))

        if n % 10 == 0:
            # iteration accuracy
            dev_predictions = frozen_predict(parallel_model, device, dev_ab, dev_ba,
                                             batch_size, lm_output_file_path_dev, force_lm_output)
            dev_predictions = torch.squeeze(dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))
            scorer_folder = working_folder + f'/scorer_frozen/chk_{n}'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)
            model_path = scorer_folder + '/linear.chkpt'
            torch.save(parallel_model.module.linear.state_dict(), model_path)
            parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
            parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/scorer_frozen/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')


def train_triplet(train_pairs,
          #train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          working_folder,
          device,
          batch_size=32,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    triplet_loss = (
    nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - cos(x, y), margin=1))
    
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, 
    
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer
    print("endID", parallel_model.module.end_id)
    print("startID", parallel_model.module.start_id)
    print("DOC endID", parallel_model.module.docend_id)
    print("DOC startID", parallel_model.module.docstart_id)
    


    dev_labels = torch.LongTensor(dev_labels)
 
    
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_sentence")
        

    new_batch_size = batch_size
    chunk_size =5000
    print("batch size",new_batch_size )
    print("chunk size",chunk_size )
    np.random.seed(42)
    for n in range(n_iters):
        
        train_pairs = list(np.array(train_pairs)[np.random.choice(len(train_pairs), size=5000, replace=False)])
        
        train_indices = list(range(len(train_pairs)))
        
        random.Random(42).shuffle(train_indices)
      
        train_pairs = list((train_pairs[i] for i in train_indices))
      
        iteration_loss = 0.
        for j in tqdm(range(0, len(train_indices), chunk_size), desc='Creating batches for tokenizaton'):
    
            chunk_train_indices = train_indices[j: j + chunk_size]
            
            chunk_train_pairs= train_pairs[j: j + chunk_size]
           
            batch_indices = list(range(len(chunk_train_pairs)))
       
            train_aa, train_ab, train_ac = tokenize_triplets(tokenizer, chunk_train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_sentence")

        
            #new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
            for i in tqdm(range(0, len(chunk_train_indices), new_batch_size), desc='Training'):


                optimizer.zero_grad()

                dev_pairs = dev_pairs[i: i + new_batch_size]

                mini_batch_indices = batch_indices[i: i + new_batch_size]
              
                tensor_aa = forward_ab(parallel_model, train_aa, device, mini_batch_indices)
                tensor_ab = forward_ab(parallel_model, train_ab, device, mini_batch_indices)
                tensor_ac = forward_ab(parallel_model, train_ac, device, mini_batch_indices)
      

                #put weight on the loss function, or for the encoder model just have the bce for labels as well as commutative scores 

                
                loss = triplet_loss(tensor_aa,tensor_ab, tensor_ac)
                print("sample loss", loss)
                loss.backward()
                
                
#                 loss = bce_loss(torch.squeeze(scores_mean), batch_labels)  

#                 loss.backward()

                optimizer.step()

                iteration_loss += loss.item()

                #del tensor_aa,tensor_ab,tensor_ac

            print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
            # iteration accuracy

            #dev_predictions = predict(parallel_model, device, dev_ab, dev_ba, batch_size)
            dev_predictions = cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size)
         
            dev_predictions = torch.squeeze(dev_predictions)
            print("Dev labels", dev_labels)
            print("Dev predictions", dev_predictions)

            print("dev accuracy:", accuracy(dev_predictions, dev_labels))
            print("dev precision:", precision(dev_predictions, dev_labels))
            print("dev f1:", f1_score(dev_predictions, dev_labels))

        scorer_folder = working_folder + f'/scorer_tn_fn_triplet/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear.chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/scorer_tn_fn_triplet/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
    
    return tensor_aa, tensor_ab,tensor_ac
def train_cross(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          working_folder,
          device,
          batch_size=32,
          n_iters=10,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineEmbeddingLoss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #try SGD with weight decay, l-2 penalty, momentum 0.9, 0.09, 0.05, Doesn't work
    
    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])
    
#     optimizer = torch.optim.SGD([
#         {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
#         {'params': parallel_model.module.linear.parameters(), 'lr': lr_class, 'momentum':0.09}
#     ])
    
     

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    #train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id)
    #dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id)
    
    #new tokenization for increasing context window using bert documents 
    
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    
    
    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    new_batch_size = batch_size
    print("batch size",new_batch_size )
    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        
        #new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

#             scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
#             scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)
            
                #trying the cosine embedding loss
            cos_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            cos_ba = forward_ab(parallel_model, train_ba, device, batch_indices)
            print(cos_ab.size(),cos_ba.size() )
            cos_sim = cos(cos_ab, cos_ba)
            

        

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()
            
            #del scores_ab,scores_ba
            del cos_ab,cos_ba

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        
        #dev_predictions = predict(parallel_model, device, dev_ab, dev_ba, batch_size)
        dev_predictions = cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size)
        dev_predictions = torch.squeeze(dev_predictions)
        print("dev prediction", dev_predictions)
        print("dev labels", dev_labels)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

#         scorer_folder = working_folder + f'/scorer_cross_long_cos_mse/chk_{n}'
#         if not os.path.exists(scorer_folder):
#             os.makedirs(scorer_folder)
#         model_path = scorer_folder + '/linear.chkpt'
#         torch.save(parallel_model.module.linear.state_dict(), model_path)
#         parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#         parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

#     scorer_folder = working_folder + '/scorer_cross_long_cos_mse/'
#     if not os.path.exists(scorer_folder):
#         os.makedirs(scorer_folder)
#     model_path = scorer_folder + '/linear.chkpt'
#     torch.save(parallel_model.module.linear.state_dict(), model_path)
#     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
#     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')





def batching(n, batch_size, min_batch):
    new_batch_size = batch_size
    while n % new_batch_size < min_batch:
        new_batch_size -= 1
    return new_batch_size


def predict_trained_model(model_name, linear_weights_path, test_set_path, working_folder):
    #test_pairs, test_labels = zip(*load_data(test_set_path)) 
    test_pairs, test_labels = zip(*load_data_cross_full(test_set_path)) 
    
    test_labels = torch.LongTensor(test_labels).bool().unsqueeze(1)
    triv_dev_path = parent_path + '/parsing/ecb/trivial_non_trivial_dev.csv'

#     test_pairs = test_pairs[0:50]
    
#     test_labels = test_labels[0:50]
    
    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    mention_map = ecb_mention_map
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    device = torch.device('cuda:0')
    print(device)
    #device_ids = list(range(4))
    device_ids = [0]
    print(model_name)
    print(linear_weights_path)
    linear_weights = torch.load(linear_weights_path)
    scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name,
                                           linear_weights=linear_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id)

    predictions = predict(parallel_model, device, test_ab, test_ba, batch_size=100)
    #predictions = predictions.long()
    #print(predictions, test_labels)
    #print(len(predictions))
    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))
    return predictions, test_labels,test_pairs 

def predict_triv_model(model_name, linear_weights_path, test_pairs,mention_map, working_folder):
    #test_pairs, test_labels = zip(*load_data(test_set_path)) 
#     test_pairs, test_labels = zip(*load_data_cross_full(test_set_path)) 
    
#     test_labels = torch.LongTensor(test_labels).bool().unsqueeze(1)
#     triv_dev_path = parent_path + '/parsing/ecb/trivial_non_trivial_dev.csv'

#     test_pairs = test_pairs[0:50]
    
#     test_labels = test_labels[0:50]
    
#     # read annotations
#     ecb_mention_map_path = working_folder + '/mention_map.pkl'
#     if not os.path.exists(ecb_mention_map_path):
#         parse_annotations(ann_dir, working_folder)
#     ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
#     mention_map = ecb_mention_map
#     for key, val in ecb_mention_map.items():
#         val['mention_id'] = key

    device = torch.device('cuda:0')
    print(device)
    #device_ids = list(range(4))
    device_ids = [0]
    print(model_name)
    print(linear_weights_path)
    linear_weights = torch.load(linear_weights_path)
    scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name,
                                           linear_weights=linear_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id)
    
    #change the predict here to hetonly one directional similarity scores and not labels !
    
    predictions_ab, predictions_ba = predict_cross_scores(parallel_model, device, test_ab, test_ba, batch_size=130)
    #predictions = predictions.long()
    #print(predictions, test_labels)
    #print(len(predictions))
#     print("Test accuracy:", accuracy(predictions, test_labels))
#     print("Test precision:", precision(predictions, test_labels))
#     print("Test recall:", recall(predictions, test_labels))
#     print("Test f1:", f1_score(predictions, test_labels))
    return predictions_ab, predictions_ba

def predict_trained_model_tp_fp(model_name, parallel_model, test_set_path, working_folder):
    #test_pairs, test_labels = zip(*load_data(test_set_path)) 
   # test_pairs, test_labels = zip(*load_data_cross_full(test_set_path)) 
    test_pairs, test_labels = zip(*load_data_tp_fp(test_set_path)) # simply testing on the dev set 
    
    print(len(test_pairs))
    
    #test_labels = torch.LongTensor(test_labels).bool().unsqueeze(1)
    
    
    test_pos = []
    test_neg = []

    for i, j in zip(test_pairs, test_labels):
        if j==1:
            test_pos.append(i)
        else:
            test_neg.append(i)

    test_pairs=test_pos[:4145] + test_neg 
    test_labels= [1]*4145 + [0]*len(test_neg )
    
    test_labels = torch.LongTensor(test_labels)

 
    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    mention_map = ecb_mention_map
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    device = torch.device('cuda:0')
    print(device)
    #device_ids = list(range(4))
    device_ids = [0]
    
    
    
    
    #         scorer_folder = working_folder + f'/scorer_tp_fp_bce/chk_{n}'
#         if not os.path.exists(scorer_folder):
#             os.makedirs(scorer_folder)
#         model_path = scorer_folder + '/linear.chkpt'
#         torch.save(parallel_model.module.linear.state_dict(), model_path)
#     model_name = working_folder + '/scorer_tp_fp_bce/chk_13/'
#     print(model_name)
#     linear_weights_path =  model_name + '/linear.chkpt'
#     print(linear_weights_path)
#     linear_weights = torch.load(linear_weights_path)
#     scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name,
#                                            linear_weights=linear_weights).to(device)
#     parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
#     parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    #test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id)
    test_ab, test_ba  = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_sentence")

    predictions = cos_align_predict(parallel_model, device, test_ab, test_ba, batch_size=500)
    predictions = torch.squeeze(predictions)
    
    #predictions = predictions.long()
    #print(predictions, test_labels)
    #print(len(predictions))
    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))
    return predictions, test_labels,test_pairs

if __name__ == '__main__':
    


#training the first classifier! Triviality Detector
   
    
    working_folder = parent_path + "/parsing/ecb"
    scorer_folder = working_folder +'/scorer_tn_fn_triplet/chk_3/'
    model_name = scorer_folder + 'bert'
    linear_weights_path = scorer_folder + 'linear.chkpt'

    #for false neg and true neg training
    
#     triv_train_path = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_train.tsv'
    triv_dev_path = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_dev.tsv'
    triv_test_path = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_test.tsv'
   

    with open(triv_train_path) as tff:
        rows = [line.strip().split('\t') for line in tff]

    samples = defaultdict(dict)
    for m1, m2, label in rows:
        if label not in samples[m1]:
            samples[m1][label] = []
        samples[m1][label].append((m1, m2, label))

    samples_true = {key: val for key, val in samples.items() if len(val) > 1}
    print(len(rows))

    triplets = [(a, b,c) for a in samples_true for _, b, _ in samples_true [a]['POS'] for _, c, _ in  samples_true[a] ['NEG']]
    
    train_pairs = triplets

    train_pairs, train_labels = zip(*load_data(triv_train_path))
    
    #load train and dev data for TP FP examples
#     train_pairs, train_labels= zip(*load_data_tp_fp(triv_train_path))
    dev_pairs, dev_labels= zip(*load_data_tp_fp(triv_dev_path))

    print(len(dev_pairs))
    
    
    
    
    device = torch.device('cuda:0')
    #model_name = 'allenai/longformer-base-4096'
    #model_name = 'biu-nlp/cdlm'
    linear_weights = torch.load(linear_weights_path)
    scorer_module = LongFormerTriplet(is_training=False, model_name=model_name, linear_weights=linear_weights).to(device)
    #scorer_module = LongFormerCrossEncoder(is_training=True, model_name=model_name).to(device)
    #scorer_module = LongFormerTriplet(is_training=True, model_name=model_name).to(device)
    
    
    
   

#     #device_ids = list(range(2))
    device_ids = [0]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    #print(parallel_model)
    working_folder = parent_path + "/parsing/ecb"
    
    predictions, test_labels,test_pairs  = predict_trained_model_tp_fp(model_name, parallel_model, triv_test_path, working_folder)
    print(len(predictions))
    
    pickle.dump(predictions, open(working_folder + '/tn_fn_test_predictions', 'wb'))
    pickle.dump(test_labels, open(working_folder + '/tn_fn_test_labels', 'wb'))
    pickle.dump(test_pairs, open(working_folder + '/tn_fn_test_pairs', 'wb'))


    

    
    tensor_aa, tensor_ab,tensor_ac = train_triplet(train_pairs,
                 #train_labels,
                 dev_pairs,
                 dev_labels,
                 parallel_model,
                 ecb_mention_map,
                 working_folder,
                 device, batch_size=20, lr_class=0.00001, lr_lm=0.00001,
                 # force_lm_output=False,
                 n_iters=20)
    
    
    
    
    
    

 





