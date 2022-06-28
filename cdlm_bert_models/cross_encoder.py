#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *

from transformers import (AutoModelWithLMHead, 
                          AutoTokenizer, 
                          BertConfig)

import configparser


config_model = configparser.ConfigParser()
config_model.read('/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/model_parameters.ini')

#config_model.read('../model_parameters.ini')


# coref_model_path ='/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/coreference/coref_bert_base'
# coref_model_path ='/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/coreference/coref_bert_base'
# bert_large_path = "/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/coreference/bert_large"
# bert_large_tokenizer_path = '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/coreference/bert_large'
# coref_bert_tokenizer_path = '/s/chopin/d/proj/ramfis-aida/coreference_and_annotations/coreference/coref_bert_base/coref_token'



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

class FullCrossEncoder(nn.Module):
    def __init__(self, config, is_training=True, long=False, coref_model = 'bert_base'):
        super(FullCrossEncoder, self).__init__()
        print(coref_model)
      
        #self.segment_size = config.segment_window * 2
        #self.tokenizer = AutoTokenizer.from_pretrained(coref_bert_tokenizer_path)
        
        if coref_model == 'bert_base':

            self.tokenizer = BertTokenizer.from_pretrained(config_model['model_bert_base']['token_path'])
            self.model = BertModel.from_pretrained(config_model['model_bert_base']['model_name'])
        elif coref_model == 'bert_large':
            
            self.tokenizer = BertTokenizer.from_pretrained(config_model['model_bert_large']['token_path'])
            self.model = BertModel.from_pretrained(config_model['model_bert_large']['model_name'])
        elif coref_model == 'bert_coref':
            self.tokenizer = AutoTokenizer.from_pretrained(config_model['model_bert_coref']['token_path'])
            self.model = AutoModel.from_pretrained(config_model['model_bert_coref']['model_name'])
        #model = BertModel.from_pretrained("bert-large-uncased")
        #print(config_model['model_bert_base']['model_name'])
        #self.tokenizer = BertTokenizer.from_pretrained(bert_large_tokenizer_path)
        


        #self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens = True)
        #self.tokenizer.add_tokens(['<g>'], special_tokens = True)
        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.vals = [self.start_id, self.end_id]
        self.long = long
#         if not is_training and config.pretrained_model:
#             self.model = AutoModel.from_pretrained(config.cdlm_path)
#         else:
            #self.model = AutoModel.from_pretrained(coref_model_path)
            #self.model = BertModel.from_pretrained(config_model['model_bert_base']['model_name'])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size
        #self.tokenizer.save_pretrained("./models/bert_large_tokenizer/")
        #self.tokenizer2 = BertModel.from_pretrained("./models/bert_large_tokenizer/")
        if not self.long:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        self.linear.apply(init_weights)



    def forward(self, input_ids, attention_mask=None, arg1=None, arg2=None):
        output, _ = self.model(input_ids, attention_mask=attention_mask)
        if self.long:
            arg1_vec = (output*arg1.unsqueeze(-1)).sum(1)
            arg2_vec = (output*arg2.unsqueeze(-1)).sum(1)
        cls_vector = output[:, 0, :]
        if not self.long:
            scores = self.linear(cls_vector)
        else:
            scores = self.linear(torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec*arg2_vec],dim=1))
        return scores

    def generate_rep(self, input_ids, attention_mask=None, arg1=None,):
        output, _ = self.model(input_ids, attention_mask=attention_mask)
        arg1_vec = (output*arg1.unsqueeze(-1)).sum(1)
        return arg1_vec


class FullCrossEncoderSingle(FullCrossEncoder):
    def __init__(self, config, is_training=True, long=False, coref_model='bert_large', *args, **kwargs):
        super(FullCrossEncoderSingle, self).__init__(config, is_training=is_training, long=long,coref_model=coref_model )
        self.model_type = coref_model

    def forward(self, input_ids, attention_mask=None, arg1=None, arg2=None):
        output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
         
        arg1_vec = (output * arg1.unsqueeze(-1)).sum(1)
        return arg1_vec


# In[ ]:




