import torch.nn as nn
import torch
from transformers import *
import numpy as np
import pyhocon
import os

# relative path of config file

config_file_path = os.path.dirname(__file__) + '/../cdlm/config_pairwise_long_reg_span.json'
config = pyhocon.ConfigFactory.parse_file(config_file_path)
print(config.cdlm_path)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class FullCrossEncoder(nn.Module):
    def __init__(self, config, is_training=True, long=False):
        super(FullCrossEncoder, self).__init__()
        self.segment_size = config.segment_window * 2
        self.tokenizer = LongformerTokenizer.from_pretrained(config.cdlm_path)
        self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
        self.tokenizer.add_tokens(['<g>'], special_tokens=True)
        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.vals = [self.start_id, self.end_id]
        self.long = long
        if not is_training and config.pretrained_model:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
        else:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size
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
        output = self.model(input_ids, attention_mask=attention_mask)

        arg1_vec = (output[0]*arg1.unsqueeze(-1)).sum(1)
        arg2_vec = (output[0]*arg2.unsqueeze(-1)).sum(1)
        # cls_vector = output[:, 0, :]
        # changed by Abhijnan to catch the tuple output in the right format, i.e. first element
        cls_vector = output[0][:, 0, :]
        if not self.long:
            scores = self.linear(cls_vector)
        else:
            scores = self.linear(torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec*arg2_vec], dim=1))
        # return output #debugging
        return scores

    def generate_rep(self, input_ids, attention_mask=None, arg1=None,):
        output, _ = self.model(input_ids, attention_mask=attention_mask)
        arg1_vec = (output*arg1.unsqueeze(-1)).sum(1)
        return arg1_vec


class FullCrossEncoderSingle(nn.Module):
    """
    This module is derived from FullCrossEncoder.
    The purpose of this module is to be able to generate the embedding for a mention individually with CDLM
    """
  
    def __init__(self, config, is_training=True, long=False):
        super(FullCrossEncoderSingle, self).__init__()
        self.segment_size = config.segment_window * 2
        self.tokenizer = LongformerTokenizer.from_pretrained(config.cdlm_path)
        self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
        self.tokenizer.add_tokens(['<g>'], special_tokens=True)
        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.vals = [self.start_id, self.end_id]
        self.long = long
        if not is_training and config.pretrained_model:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
        else:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, arg, attention_mask):
        """

        Parameters
        ----------
        input_ids: LongTensor
            tokenized ids of the sentence/document

        arg: LongTensor
            tensor that indicates the word pieces for the mentions in the sentence/document

        attention_mask: LongTensor
            attention mask

        Returns
        -------
        FloatTensor
            The average vector of the mention word pieces

        """
        output = self.model(input_ids, attention_mask=attention_mask)
        arg_vec = (output[0] * arg.unsqueeze(-1)).sum(1)
        return arg_vec


class Regressor(nn.Module):
    """
    NN module of the regressor over similarity features
    """
    def __init__(self, feature_len):
        """

        Parameters
        ----------
        feature_len
        """
        super(Regressor, self).__init__()
        self.hidden_size = feature_len
        self.second_layer = 4

        self.linear1 = torch.nn.Linear(self.hidden_size, self.second_layer)
        self.linear2 = torch.nn.Linear(self.second_layer, 1)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        stdv2 = 1. / np.sqrt(self.linear1.weight.shape[0])
        self.linear1.weight.data.uniform_(0., stdv2)
        self.linear2.weight.data.uniform_(0., 1. / np.sqrt(self.linear2.weight.shape[0]))
        torch.nn.init.constant_(self.linear1.bias.data, 0.)
        torch.nn.init.constant_(self.linear2.bias.data, 0.)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out
