import os.path

from sklearn.model_selection import train_test_split
import pyhocon
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


def load_data(trivial_non_trivial_path):
    all_examples = []
    with open(trivial_non_trivial_path) as tnf:
        for line in tnf:
            row = line.strip().split(',')
            mention_pair = row[:2]
            triviality_label = int(row[2])
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


def tokenize(tokenizer, mention_pairs, mention_map):
    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
        sentence_a = mention_map[m1]['bert_sentence']
        sentence_b = mention_map[m2]['bert_sentence']

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end, doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    tokenized_ab = tokenizer(pairwise_bert_instances_ab, pad_to_max_length=True, add_special_tokens=False,
                             truncation=False)
    tokenized_ba = tokenizer(pairwise_bert_instances_ba, pad_to_max_length=True, add_special_tokens=False,
                             truncation=False)

    return (torch.LongTensor(tokenized_ab['input_ids']), torch.LongTensor(tokenized_ab['attention_mask'])), (
    torch.LongTensor(tokenized_ba['input_ids']), torch.LongTensor(tokenized_ba['attention_mask']))


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


def predict(parallel_model, device, tensor_ab, tensor_ba, am_ab, am_ba, batch_size):
    n = tensor_ab.shape[0]
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_tensor_ab = tensor_ab[i:i + batch_size, :]
            batch_tensor_ba = tensor_ba[i:i + batch_size, :]

            batch_am_ab = am_ab[i:i + batch_size, :]
            batch_am_ba = am_ba[i:i + batch_size, :]

            am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)
            am_g_ba, arg1_ba, arg2_ba = get_arg_attention_mask(batch_tensor_ba, parallel_model)

            batch_tensor_ab.to(device)
            batch_tensor_ba.to(device)

            scores_ab = parallel_model(batch_tensor_ab, attention_mask=batch_am_ab,
                                       global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab)
            scores_ba = parallel_model(batch_tensor_ba, attention_mask=batch_am_ba,
                                       global_attention_mask=am_g_ba, arg1=arg1_ba, arg2=arg2_ba)

            scores_mean = (scores_ab + scores_ba) / 2

            batch_predictions = (scores_mean > 0.5).detach().cpu()
            predictions.append(batch_predictions)

    return torch.cat(predictions)


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

    # all_examples = load_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    (train_tensor_ab, train_am_ab), (train_tensor_ba, train_am_ba) = tokenize(tokenizer, train_pairs, mention_map)
    (dev_tensor_ab, dev_am_ab), (dev_tensor_ba, dev_am_ba) = tokenize(tokenizer, dev_pairs, mention_map)

    print(train_tensor_ab.shape)
    print(train_am_ab.shape)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)

    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        for i in tqdm(range(0, len(train_indices), batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + batch_size]

            batch_tensor_ab, batch_tensor_ba = (
                train_tensor_ab[batch_indices, :], train_tensor_ba[batch_indices, :]
            )

            batch_am_ab, batch_am_ba = (
                train_am_ab[batch_indices, :], train_am_ba[batch_indices, :]
            )

            am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)
            am_g_ba, arg1_ba, arg2_ba = get_arg_attention_mask(batch_tensor_ba, parallel_model)

            batch_labels = train_labels[batch_indices].to(device)
            batch_tensor_ab.to(device)
            batch_tensor_ba.to(device)

            scores_ab = parallel_model(batch_tensor_ab, attention_mask=batch_am_ab,
                                       global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab)
            scores_ba = parallel_model(batch_tensor_ba, attention_mask=batch_am_ba,
                                       global_attention_mask=am_g_ba, arg1=arg1_ba, arg2=arg2_ba)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(torch.squeeze(scores_mean), batch_labels) + mse_loss(scores_ab, scores_ba)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_predictions = predict(parallel_model, device, dev_tensor_ab, dev_tensor_ba, dev_am_ab, dev_am_ba,
                                  batch_size)
        dev_predictions = torch.squeeze(dev_predictions)
        print(dev_predictions.shape)
        print(accuracy(dev_predictions, dev_labels))
        print(f1_score(dev_predictions, dev_labels))

        scorer_folder = working_folder + f'/scorer/chk_{n}'
        if not os.path.exists(scorer_folder):
            os.makedirs(scorer_folder)
        model_path = scorer_folder + '/linear.chkpt'
        torch.save(parallel_model.module.linear.state_dict(), model_path)
        parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        parallel_model.module.model.tokenizer.save_pretrained(scorer_folder + '/bert')

    scorer_folder = working_folder + '/scorer/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')


if __name__ == '__main__':
    triv_train_path = '../parsing/ecb/trivial_non_trivial_train.csv'
    triv_dev_path = '../parsing/ecb/trivial_non_trivial_dev.csv'

    train_pairs, train_labels = zip(*load_data(triv_train_path))
    dev_pairs, dev_labels = zip(*load_data(triv_dev_path))

    device = torch.device('cuda:0')

    scorer_module = LongFormerCrossEncoder(is_training=True).to(device)
    device_ids = [0, 1]

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    working_folder = "../parsing/ecb"
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
          device, batch_size=32, lr_class=0.00001, lr_lm=0.000001,
          n_iters=100)
