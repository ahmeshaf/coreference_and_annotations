import os
import sys
import csv
parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(parent_path)

import os.path
import pickle

from sklearn.model_selection import train_test_split
from coreference.models import LongFormerCrossEncoder
import torch
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


def load_easy_hard_data(trivial_non_trivial_path):
    all_examples = []
    label_map = {'HARD': 0, 'EASY': 1}
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            triviality_label = label_map[row[2]]
            all_examples.append((mention_pair, triviality_label))

    return all_examples
#load lemma balanced TP and FP tsv pairs

def load_data_tp_fp(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        rd = csv.reader(tnf, delimiter="\t", quotechar='"')
        
        for line in rd:
          
            mention_pair = line[:2]
          
            if line[2] =='POS':
                triviality_label = 1
                all_examples.append((mention_pair, triviality_label))
                
            else:
                triviality_label = 0
                all_examples.append((mention_pair, triviality_label))
          
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
    """
    Tokenized instances in both directions for bert_sentence or bert_doc as context 
    """
    if max_sentence_len is None:
        
        max_sentence_len = tokenizer.model_max_length #try 512 here, 

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
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

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
                in_truncated = input_id[m_end_index-(max_sentence_len//4): m_end_index] + \
                               input_id[m_end_index: m_end_index + (max_sentence_len//4)]


                in_truncated = in_truncated + [tokenizer.pad_token_id]*(max_sentence_len//2 - len(in_truncated))
                input_ids_truncated.append(in_truncated)
        else:
            
         
            for input_id in input_ids:   #truncate both the tokens from the input IDs after tokenization 
                #ensure that the paired mentions with the special tokens are present in the final truncated tensors. 
                
             
                global_input_id = [1]  # token ID for global attention 
                m_start_index = input_id.index(m_start)  # start ID 
                m_end_index = input_id.index(m_end) # end ID
                doc_start_token = [50267]  # document start ID 
                doc_end_token =  [50268]  # document end ID
                in_truncated = input_id[m_start_index-(max_sentence_len//4): m_start_index] + \
                input_id[m_start_index:m_end_index+1] + input_id[ m_end_index+1: m_end_index + (max_sentence_len//4)]

                in_truncated = global_input_id + doc_start_token + in_truncated + doc_end_token # add global attention on CLS
                
                
                in_truncated = in_truncated + [tokenizer.pad_token_id]*((max_sentence_len//2) - len(in_truncated))
                
                input_ids_truncated.append(in_truncated[0:256])   

#                 new_ids = torch.cat((doc_start_token , in_truncated), dim=1)
#                 new_ids = torch.cat((global_input_id, new_ids), dim=1)
#                 new_ids = torch.cat((new_ids, doc_end_token), dim=1)
                #in_truncated = new_ids 
# #                  
# #                               

                
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
    #print("non zero counts", v.nonzero().shape)
    #print(k.shape,p.shape)
    #m = m_start_indicator + m_end_indicator
    #v = (m_start_indicator.int() + m_end_indicator.int()).bool()

    # non-zero indices are the tokens corresponding to <m> and </m>
    #nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))
 
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
            scores_mean = (scores_ab + scores_ba) / 2
            batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)

def cos_align_predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
     
    #new_batch_size = batching(n, batch_size, len(device_ids))
    #batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            cos_ab = forward_ab(parallel_model, dev_ab, device, batch_indices) # these scores are actually embeddings 
            cos_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            batch_predictions = cos(cos_ab, cos_ba).detach().cpu()
           # print("batch dev prediction", batch_predictions)

         
#             batch_predictions = (cos_sim > 0.7).detach().cpu()
            
            condition = (batch_predictions>0.8).detach().cpu()
            #print(condition.size())
            batch_predictions = batch_predictions.where(~condition, torch.tensor(0.0))
            batch_predictions = batch_predictions.where(condition, torch.tensor(1.0))
 
            predictions.append(batch_predictions)

    return torch.cat(predictions)

def predict(parallel_model, device, dev_ab, dev_ba, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    new_batch_size = batching(n, batch_size, len(device_ids))
    batch_size = new_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)

            scores_mean = (scores_ab + scores_ba) / 2

            batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)


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


def train(train_pairs,
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

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)

    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels) + mse_loss(scores_ab, scores_ba)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()
            
            del scores_ab,scores_ba

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_predictions = predict(parallel_model, device, dev_ab, dev_ba, batch_size)
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

        scorer_folder = working_folder + f'/scorer/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear.chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/scorer/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

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
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map,parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.start_id, parallel_model.module.end_id, context = "bert_doc")

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    new_batch_size = batch_size
    #print("batch size",new_batch_size )
    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        
        #new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

             #trying the cosine embedding loss
            cos_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            cos_ba = forward_ab(parallel_model, train_ba, device, batch_indices)
            
            cos_sim = cos(cos_ab, cos_ba)
            

            batch_labels = train_labels[batch_indices].to(device)
          
            
            loss = cos_loss(cos_ab, cos_ba, batch_labels) 
        
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
#         print("dev prediction", dev_predictions)
#         print("dev labels", dev_labels)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

        scorer_folder = working_folder + f'/scorer_cross_long_cos_mse/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear.chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/scorer_cross_long_cos_mse/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')

    
def batching(n, batch_size, min_batch):
    new_batch_size = batch_size
    while n % new_batch_size < min_batch != 1:
        new_batch_size -= 1
    return new_batch_size


def predict_trained_model(model_name, linear_weights_path, test_set_path, working_folder):
    test_pairs, test_labels = zip(*load_easy_hard_data(test_set_path))
    test_labels = torch.LongTensor(test_labels)
    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    device = torch.device('cuda:0')
    device_ids = list(range(4))
    linear_weights = torch.load(linear_weights_path)
    scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name,
                                           linear_weights=linear_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id)

    predictions = predict(parallel_model, device, test_ab, test_ba, batch_size=128)
    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))


if __name__ == '__main__':

    triv_train_path = parent_path + '/parsing/ecb/trivial_non_trivial_train.csv'
    triv_dev_path = parent_path + '/parsing/ecb/trivial_non_trivial_dev.csv'

    train_pairs, train_labels = zip(*load_easy_hard_data(triv_train_path))
    dev_pairs, dev_labels = zip(*load_easy_hard_data(triv_dev_path))

    train_pairs = list(train_pairs)
    train_labels = list(train_labels)

    device = torch.device('cuda:0')
    model_name = 'allenai/longformer-base-4096'
    
    
    # for triviality detector encoder with bce and mse loss
    
    scorer_module = LongFormerCrossEncoder(is_training=True, model_name=model_name).to(device)
    
    # for cosine embedding alignment cross encoder second classifier
    #scorer_module = LongFormerCosAlign(is_training=True, model_name=model_name).to(device)

    device_ids = list(range(1))

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    working_folder = parent_path + "/parsing/ecb"

    # edit this or not!
    ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"

    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    if not os.path.exists(ecb_mention_map_path):
        parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          ecb_mention_map,
          working_folder,
          device, batch_size=2, lr_class=0.0001, lr_lm=0.000001,
          # force_lm_output=False,
          n_iters=100)
