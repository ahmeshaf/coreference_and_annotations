
# Author: Rehan and Abhijnan
import pickle
import pyhocon
import torch.nn.functional as F
from models import *
import torch
from tqdm import tqdm


def generate_cdlm_embeddings_from_model(parallel_model, bert_sentences, device, batch_size=150):
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


def generate_cdlm_embeddings(mention_map, vec_map_path, key_name='bert_doc',
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
    
    men_ids, bert_sentences = zip(*[(men_id, men[key_name]) for men_id, men in mention_map.items()])
    config_file_path = '../cdlm/config_pairwise_long_reg_span.json'
    config = pyhocon.ConfigFactory.parse_file(config_file_path)
    device_ids = config.gpu_num[:num_gpus]
    device = torch.device("cuda:{}".format(device_ids[0]))
    if cpu:
        device = torch.device("cpu")

    # instantiate the cross encoder from models.py
    cross_encoder = FullCrossEncoderSingle(config, long=True)
    cross_encoder = cross_encoder.to(device)

    parallel_model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
    parallel_model.module.to(device)
   
    # get all the vectors along with their mention IDs and pickle them
    all_vectors = generate_cdlm_embeddings_from_model(parallel_model, bert_sentences, device, batch_size)
    vec_map = {men_id: vec for men_id, vec in zip(men_ids, all_vectors)}
    pickle.dump(vec_map, open(vec_map_path, 'wb'))
