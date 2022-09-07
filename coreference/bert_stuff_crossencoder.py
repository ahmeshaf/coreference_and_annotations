# Author: Rehan and Abhijnan
import pickle
import pyhocon
import torch.nn.functional as F
from transformers import AutoModel
from models import *
import torch
from tqdm import tqdm
from collections import defaultdict
model_dir = '/s/chopin/d/proj/ramfis-aida/coref/coreference_and_annotations/cdlm/models/cdlm2/'
lin_model_dir = '/s/chopin/d/proj/ramfis-aida/coref/coreference_and_annotations/cdlm/models/cdlm2/linear'

def generate_cdlm_embeddings_from_model_cross(parallel_model, m_id_list, device, batch_size=150):
    parallel_model.eval()
    topic_scores = []
    loc_mentions = []
    #batch_size=1
    
    
    results = []
    all_input_ids = list(m_id_list.items())
#     all_input_ids = all_input_ids[61000:-1]
    torch.cuda.empty_cache()
    with torch.no_grad():
        #for index in range(0, len(all_input_ids), batch_size):
        for index in tqdm(range(0, len(all_input_ids), batch_size), desc="generating "):
            
            batch = [y for x,y in all_input_ids[index:index+batch_size]]
    
            #batch = [y for x, y in m_id_list.items() ] #get the sentences from the pairwise m_id list 

            #batch = [m_id_list[x] for x, y in m_id_list.items() ] #get the sentences from the pairwise m_id list 
            batch = [x for xs in batch  for x in xs] #make a flat list for cdlm model inputs
             

            #print(batch)
            #print("batch creation", len(batch))


            bert_tokens = parallel_model.module.tokenizer(batch, pad_to_max_length=True,
                                                                 add_special_tokens=False, truncation=False)

            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            m = input_ids.cpu()
            #print("shape of m", m.shape)
            k = m == parallel_model.module.vals[0]
            p = m == parallel_model.module.vals[1]

            v = (k.int() + p.int()).bool()
            nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)
            loc_mentions= nz_indexes.cpu().numpy()
            #print("loc mention length", loc_mentions.shape)


            overflow_idx = []
            overflow_idx_diff = []

            for i, j in enumerate(loc_mentions):
                #if j[3]-j[0] >= 4096 and j[3]>=4096 :
                if j[3]-j[0] <= 4096 and j[3]>=4096 :
                    x = j[3]-j[0]
                    overflow_idx.append([i,x,j[0], j[3]])
                if j[3]-j[0] > 4096 and j[3]>=4096 :
                    y = j[3]-j[0]
                    overflow_idx_diff.append([i,j[1], j[3]])

            #cases where the last special token is outside of 4096 but difference is less than 4096 
            #print("overflow index", overflow_idx)
            #print("overflow index diff", overflow_idx_diff)
            # check if there is any indices for mentions that are overflowing 
            if len(overflow_idx)>=1 :
                #print(" getting overflow_idx")
                
                idx = [x[0] for x in overflow_idx]
                m_new = []
                #print("overflow index in first case", overflow_idx)
                for i, j in enumerate(overflow_idx):
                    
                    #bert_sent = bert_sentences[j[0]]
                    m_new.append(m[j[0],  j[3]-4097+1:j[3]+1])


                m_new_tensor = torch.stack(m_new) 
#                 a = m_new_tensor == parallel_model.module.vals[0]
#                 b = m_new_tensor == parallel_model.module.vals[1]
#                 c = (a.int() + b.int()).bool()
#                 print("non zero first case for m new tensor 4 before padding?", c.nonzero()[:, 1].shape)
                m_new_tensor = F.pad(m_new_tensor, pad=(0, m.shape[1]-m_new_tensor.shape[1]), value=1)
#                 a = m_new_tensor == parallel_model.module.vals[0]
#                 b = m_new_tensor == parallel_model.module.vals[1]
#                 c = (a.int() + b.int()).bool()
#                 print("non zero first case for m new tensor 4?", c.nonzero()[:, 1].shape)
                
            

                m[idx] = m_new_tensor
              
                
                #cases where diff is more than 4096 and last token is > 4096 
                
            if len(overflow_idx_diff)>=1:
                #print(" getting overflow_idx_diff")
                idx_diff = [x[0] for x in overflow_idx_diff]
                m_new_start = []
                m_new_end = []
                for i, j in enumerate(overflow_idx_diff):

                    if j[1] >=2048:
                        m_new_start.append(m[j[0],  j[1]-2048+1:j[1]+1])
                    else:
                        m_new_start.append(m[j[0], 0:2048])

                    m_new_end.append(m[j[0],  j[2]-2048+1:j[2]+1])
                start_tensor = torch.stack(m_new_start) 
                end_tensor = torch.stack(m_new_end) 
                concat_tensor= torch.cat([start_tensor, end_tensor], dim=1)
                concat_tensor = F.pad(concat_tensor, pad=(0, m.shape[1]-concat_tensor.shape[1]), value=1)
                #print("shape of concat tensor", concat_tensor.shape)
                
                #concat_tensor= torch.cat([concat_tensor, m_new_tensor], dim=0) #concat the diff(>4096) between mentions along with no diff
                #idx_comb = idx_diff + idx

                m[idx_diff] = concat_tensor
#                 a = m == parallel_model.module.vals[0]
#                 b = m == parallel_model.module.vals[1]
#                 c = (a.int() + b.int()).bool()
#                 print("non after diff shape", c.nonzero()[:, 1].shape)
                
            
            
            k = m == parallel_model.module.vals[0]
            p = m == parallel_model.module.vals[1]
            v = (k.int() + p.int()).bool()
            #print(" shape of boolean v", v.shape)
            nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)

            q = torch.arange(m.shape[1])
            q = q.repeat(m.shape[0], 1)

            msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
            msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
            msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
            msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

            msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
            msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
            msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
            msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

            attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()

            input_ids = input_ids[:, :4096]
            attention_mask = attention_mask[:, :4096]
            attention_mask_g = attention_mask_g[:, :4096]

            # 1 because we are letting this token be attended globally,
            # not 2 because longformer issue resolved
            # https://github.com/huggingface/transformers/issues/7015
            attention_mask[:, 0] = 1
            attention_mask[attention_mask_g == 1] = 1

            arg1 = msk_0_ar.int() * msk_1_ar.int()
            arg2 = msk_2_ar.int() * msk_3_ar.int()
            arg1 = arg1[:, :4096]
            arg2 = arg2[:, :4096]
            arg1 = arg1.to(device)
            arg2 = arg2.to(device)

            scores = parallel_model(input_ids, attention_mask, arg1, arg2)
            scores = torch.sigmoid(scores)
            topic_scores.extend(scores.detach().cpu().squeeze(1).numpy())
             
    



# #     with torch.no_grad():
# #         scores = parallel_model(input_ids, attention_mask, arg1, arg2)
# #         #print(scores.shape)
# #         scores = torch.sigmoid(scores)
# #         topic_scores.extend(scores.detach().cpu().squeeze(1).numpy())

    
    # get the model outputs in batches
#     with torch.no_grad():
#         for i in tqdm(range(0, len(batch), batch_size), desc="generating "):
#             scores = parallel_model(input_ids[i:i+batch_size], attention_mask[i:i+batch_size],
#                                          arg1[i:i+batch_size], arg2[i:i+batch_size])
#             scores = torch.sigmoid(scores)
#             topic_scores.extend(scores.detach().cpu().squeeze(1).numpy())
    
    return topic_scores




def generate_cdlm_embeddings_from_model(parallel_model, m_id_list, device, batch_size=150):
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
    parallel_model.eval()

    # Store vectors here
    all_vectors = []
     
    # append indices of the special tokens around the mentions to
    # this list to find overflowing tokens(that exceed 4096 token length )
    loc_mentions = []
    
    # create batches of sentences or documents and get them tokenized
    
    
    
    
    batch = [bert_sentences[x] for x in range(len(bert_sentences))]
    bert_tokens = parallel_model.module.tokenizer(batch, pad_to_max_length=True,
                                                  add_special_tokens=False, truncation=False)
    
    input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
    attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
    m = input_ids.cpu()
    
    # get the additional special token IDs to create the binary map 
    k = m == parallel_model.module.vals[0]
    p = m == parallel_model.module.vals[1]

    v = (k.int() + p.int()).bool()
    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)
    loc_mentions.extend(nz_indexes.cpu().numpy())

    # get the overflowing token indices and chunk the document so that the special tokens are
    # within max token length(4096) for the CDLM model
    overflow_idx = []
  
    for i, j in enumerate(loc_mentions):
        if j[1] >= 4096:
            overflow_idx.append([i, j[1]])
    
    idx = [x[0] for x in overflow_idx]
    # create a list for appending the overflowing indices
    m_new = []
    for i, j in enumerate(overflow_idx):
        bert_sent = bert_sentences[j[0]]
        m_new.append(m[j[0],  j[1]-4096+1:j[1]+1])

    # convert them into a tensor and pad to original max length of the entire batch
    m_new_tensor = torch.stack(m_new) 
    m_new_tensor = F.pad(m_new_tensor, pad=(0, m.shape[1]-m_new_tensor.shape[1]), value=1)

    m[idx] = m_new_tensor
    
    # get non-zero indices for the new tensor 'm' created with mentions within 4096 tokens for entire tensor
    k = m == parallel_model.module.vals[0]
    p = m == parallel_model.module.vals[1]
    v = (k.int() + p.int()).bool()
    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 2)
    
    # create dummy tensor to get indices for the binary map creation
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)
    
    # create the  binary maps
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q

    attention_mask_g = msk_0.int() * msk_1.int()
    # get the tokens within 4096 after chunking 
    input_ids = input_ids[:, :4096]

    # 1 because we are letting this token be attended globally,
    # not 2 because longformer issue resolved
    # https://github.com/huggingface/transformers/issues/7015
    attention_mask[:, 0] = 1

    # same as above for the special mention tokens to be attended globally
    attention_mask[attention_mask_g == 1] = 1
    attention_mask = attention_mask[:, :4096]
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg1 = arg1[:, :4096]
    arg1 = arg1.to(device)

    # get the model outputs in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(bert_sentences), batch_size), desc="generating "):
            men_vectors = parallel_model(input_ids[i:i+batch_size], attention_mask[i:i+batch_size],
                                         arg1[i:i+batch_size])
            all_vectors.extend(men_vectors)
    
    return all_vectors


def generate_cross_cdlm_embeddings(mention_pairs, mention_map, vec_map_path, key_name='bert_doc',
                             num_gpus=4, batch_size=150, cpu=False):
   
    """
    Generate cdlm embeddings of the mentions in the cdlm format
    Parameters
    ----------
    mention_map: dict
        Mention map containing the documents with their mention IDs
    vec_map_path: str
        Path to store vector maps 
        
    key_name : str
        Whether shorter sentences or longer whole documents with special mention tokens
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
    
    #men_ids, bert_sentences = zip(*[(men_id, men[key_name]) for men_id, men in mention_map.items()])
    
    #create cross document bert sentences with special cdlm tokens to feed into the cdlm cross encoder model 
    cross_bert_sent = [] #cross document bert sentences
    m_id_list = defaultdict(dict)
    c=0
    for i in mention_pairs:
        #print(i)
        instances = [' '.join(['<g>', "<doc-s>",mention_map[i[0]][key_name], "</doc-s>", "<doc-s>",mention_map[i[1]][key_name],"</doc-s>"
                    
                    ])]
        

        m_id_list[i[0]+ '_'+i[1] ] = instances
        cross_bert_sent.extend(instances)
       

    config_file_path = '../cdlm/config_pairwise_long_reg_span.json'
    config = pyhocon.ConfigFactory.parse_file(config_file_path)
    device_ids = config.gpu_num[:num_gpus]
    device = torch.device("cuda:{}".format(device_ids[0]))
    if cpu:
        device = torch.device("cpu")

    # instantiate the cross encoder from models.py, use FullCrossEncoder for cross encoding mentions 
    cross_encoder = FullCrossEncoder(config, long=True, is_training=True).to(device)
    cross_encoder.model = AutoModel.from_pretrained(model_dir).to(device)
    
    cross_encoder.linear.load_state_dict(torch.load(os.path.join(lin_model_dir, 'linear')))
    cross_encoder = cross_encoder.to(device)

    parallel_model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
    parallel_model.module.to(device)
   
    # get all the vectors along with their mention IDs and pickle them
    
    
    
    #all_vectors = generate_cdlm_embeddings_from_model_cross(parallel_model, m_id_list, device, batch_size)
    scores  = generate_cdlm_embeddings_from_model_cross(parallel_model, m_id_list, device, batch_size)
    
    m_ids = [x for x in m_id_list.keys() ] #get the sentences from the pairwise m_id list 
    #m_ids = m_ids[0:50000]
    #batch = [x for xs in batch  for x in xs]
    vec_map = {men_id: vec for men_id, vec in zip(m_ids, scores)}
    pickle.dump(vec_map, open(vec_map_path, 'wb'))
    #return scores