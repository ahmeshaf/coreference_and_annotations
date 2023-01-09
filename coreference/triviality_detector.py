import os
import sys

parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
sys.path.append(parent_path)
import os.path
import pickle
from sklearn.model_selection import train_test_split
from coreference.models import LongFormerCrossEncoder, CrossEncoderTriplet
import torch
import random
from tqdm.autonotebook import tqdm
from parsing.parse_ecb import parse_annotations
from torch import nn
import numpy as np


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


def load_lemma_dataset(tsv_path, force_balance=False):
    all_examples = []
    label_map = {'POS': 1, 'NEG': 0}
    with open(tsv_path) as tnf:
        for line in tnf:
            row = line.strip().split('\t')
            mention_pair = row[:2]
            label = label_map[row[2]]
            all_examples.append((mention_pair, label))
    if force_balance:
        from collections import defaultdict
        import random
        random.seed(42)
        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        min_label = min(label2eg.keys(), key=lambda x: len(label2eg[x]))
        min_label_len = len(label2eg[min_label])

        max_eg_len = max([len(val) for val in label2eg.values()])
        random_egs = random.choices(label2eg[min_label], k=max_eg_len-min_label_len)
        all_examples.extend(random_egs)

        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        # print([len(val) for val in label2eg.values()])

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


def tokenize_triplets(tokenizer, mention_triplets, mention_map, m_end, text_id='bert_sentence', max_sentence_len=80):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    triplet_bert_instances = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (a, b, c) in mention_triplets:
        sentence_a = mention_map[a][text_id]
        sentence_b = mention_map[b][text_id]
        sentence_c = mention_map[c][text_id]

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

        instance_aa = make_instance(sentence_a, sentence_a)
        instance_ab = make_instance(sentence_a, sentence_b)
        instance_ac = make_instance(sentence_a, sentence_c)

        triplet_bert_instances.append((instance_aa, instance_ab, instance_ac))

        # pairwise_bert_instances_ab.append(instance_ab)

        # instance_ba = make_instance(sentence_b, sentence_a)
        # pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

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

    all_aa, all_ab, all_ac = zip(*triplet_bert_instances)

    tokenized_aa = ab_tokenized(all_aa)
    tokenized_ab = ab_tokenized(all_ab)
    tokenized_ac = ab_tokenized(all_ac)

    return tokenized_aa, tokenized_ab, tokenized_ac


def tokenize(tokenizer, mention_pairs, mention_map, m_end, max_sentence_len=80):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
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
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

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

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id

    m = m_start_indicator + m_end_indicator

    # non-zero indices are the tokens corresponding to <m> and </m>
    nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))

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
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()

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
            # scores_mean = (scores_ab + scores_ba) / 2
            scores_mean = scores_ab
            batch_predictions = (scores_mean > 0.5).detach().cpu()
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
            # scores_ba = parallel_model(ba_out.to(device), pre_lm_out=True)
            # scores_mean = (scores_ab + scores_ba) / 2
            scores_mean = scores_ab
            batch_labels = train_labels[batch_indices].to(device)
            loss = bce_loss(torch.squeeze(scores_mean), batch_labels)
                   # + mse_loss(scores_ab, scores_ba)
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
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id)
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id)

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

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

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

    test_ab, test_ba = tokenize(tokenizer, test_pairs, ecb_mention_map, parallel_model.module.end_id)

    predictions = predict(parallel_model, device, test_ab, test_ba, batch_size=128)
    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))


def create_train_triplets(train_pairs, train_labels):
    train_pairs_labels = [(a, b, l) for (a, b), l in zip(train_pairs, train_labels)]
    pos_train_pairs_labels = [pl for pl in train_pairs_labels if pl[-1] == 1]
    neg_train_pairs_labels = [pl for pl in train_pairs_labels if pl[-1] == 0]

    # a: anchor, b: positive sample, c: negative sample
    abc_triplets = []

    from collections import defaultdict
    a2c_pls = defaultdict(list)
    for pl in neg_train_pairs_labels:
        a2c_pls[pl[0]].append(pl)

    for a, b, _ in pos_train_pairs_labels:
        ac_pls = a2c_pls[a]
        for _, c, _ in ac_pls:
            abc_triplets.append((a, b, c))

    return abc_triplets


def predict_triplet(parallel_model, dev_aa, dev_ab, batch_size, threshold=0.9):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    predictions = []
    new_batch_size = batching(n, batch_size, len(device_ids))
    batch_size = new_batch_size
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]

            anchor = forward_ab(parallel_model, dev_aa, device, batch_indices)
            mention_b = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_mean = cosine_sim(anchor, mention_b)
            batch_predictions = (scores_mean > threshold).detach().cpu()
            predictions.append(batch_predictions)
    return torch.cat(predictions)


def train_triplet_loss(train_triplets, dev_pairs, dev_labels, mention_map, working_folder,
                       batch_size=32,
                       n_iters=10,
                       lr_lm=0.00001, lr_class=0.0001):

    model_name = 'bert-large-cased'
    scorer_module = CrossEncoderTriplet(is_training=True, model_name=model_name).to(device)

    # device_ids = list(range(1))

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    triplet_loss = nn.TripletMarginWithDistanceLoss(
                    margin=1.0,
                    distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y))

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_aa, train_ab, train_ac = tokenize_triplets(tokenizer, train_triplets, mention_map, parallel_model.module.end_id)

    # make dev pairs look like triples for evaluating
    dev_triplets = [(a, b, b) for a, b in dev_pairs]
    dev_aa, dev_ab, _ = tokenize_triplets(tokenizer, dev_triplets, mention_map, parallel_model.module.end_id)

    dev_labels = torch.LongTensor(dev_labels)

    for n in range(n_iters):
        train_indices = list(range(len(train_triplets)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]
            anchor = forward_ab(parallel_model, train_aa, device, batch_indices)
            positive = forward_ab(parallel_model, train_ab, device, batch_indices)
            negative = forward_ab(parallel_model, train_ac, device, batch_indices)
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            iteration_loss += loss.item()
        print(f'Iteration {n} Loss:', iteration_loss / len(train_indices))
        # iteration accuracy
        dev_predictions = predict_triplet(parallel_model, dev_aa, dev_ab, batch_size)
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

        scorer_folder = working_folder + f'/scorer/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear_neg .chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert_neg')
        parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert_neg')


def train_ce_neg():
    tn_fn_train_path = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_train.tsv'
    tn_fn_dev_path = parent_path + '/parsing/ecb/lemma_balanced_tn_fn_dev.tsv'

    train_pairs, train_labels = zip(*load_lemma_dataset(tn_fn_train_path))
    dev_pairs, dev_labels = zip(*load_lemma_dataset(tn_fn_dev_path))

    train_triplets = create_train_triplets(train_pairs, train_labels)

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

    dev_triplets = create_train_triplets(dev_pairs, dev_labels)

    dev_pairs_bal = []
    dev_labels_bal = []

    pos_pairs_set = set()

    for a, b, c in dev_triplets:
        if (a, b) not in pos_pairs_set:
            dev_pairs_bal.append((a, b))
            dev_labels_bal.append(1)
            dev_pairs_bal.append((a, c))
            dev_labels_bal.append(0)
            pos_pairs_set.add((a, b))

    train_triplet_loss(train_triplets, dev_pairs_bal, dev_labels_bal, ecb_mention_map, working_folder)


def train_ce_pos():
    triv_train_path = parent_path + '/parsing/ecb/lemma_balanced_tp_fp_train.tsv'
    triv_dev_path = parent_path + '/parsing/ecb/lemma_balanced_tp_fp_dev.tsv'

    train_pairs, train_labels = zip(*load_lemma_dataset(triv_train_path))
    dev_pairs, dev_labels = zip(*load_lemma_dataset(triv_dev_path))

    train_pairs = list(train_pairs)
    train_labels = list(train_labels)

    model_name = 'bert-large-cased'
    scorer_module = LongFormerCrossEncoder(is_training=False, model_name=model_name).to(device)

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

    # train(train_pairs,
    #       train_labels,
    #       dev_pairs,
    #       dev_labels,
    #       parallel_model,
    #       ecb_mention_map,
    #       working_folder,
    #       device, batch_size=2, lr_class=0.0001, lr_lm=0.000001,
    #       # force_lm_output=False,
    #       n_iters=100)
    train_frozen(train_pairs,
                 train_labels,
                 dev_pairs,
                 dev_labels,
                 parallel_model,
                 ecb_mention_map,
                 working_folder,
                 device, batch_size=128, lr_class=0.001,
                 force_lm_output=False,
                 n_iters=100)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    device_ids = list(range(1))
    train_ce_neg()

