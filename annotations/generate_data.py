import pickle
import re
from tqdm import tqdm


def generate_pair_sentences(ann_dir, working_folder, split):
    # read annotations
    ecb_mention_map_path = working_folder + '/mention_map.pkl'
    # if not os.path.exists(ecb_mention_map_path):
    #     parse_annotations(ann_dir, working_folder)
    ecb_mention_map = pickle.load(open(ecb_mention_map_path, 'rb'))
    for key, val in ecb_mention_map.items():
        val['mention_id'] = key

    curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == 'evt'}

    with open(working_folder + '/mlm_data2.txt', 'w') as lf:
        for val in curr_mention_map.values():
            bert_sentence = val['bert_sentence']
            sentence = val['sentence']
            masked_bert_sentence = re.sub('<m>.*</m>', '<m> <mask> </m>', bert_sentence)
            lf.write(sentence + '\n')
            lf.write(bert_sentence + '\n')
            lf.write(masked_bert_sentence + '\n')
        # curr_mention_map = {m_id: val for m_id, val in ecb_mention_map.items() if val['men_type'] == 'evt' and
        #                     val['split'] == split}
        for val1 in tqdm(list(curr_mention_map.values())):
            buffer = []
            for val2 in curr_mention_map.values():
                buffer.append(val1['bert_sentence'] + ' </s> ' + val2['bert_sentence'])
            lf.write('\n'.join(buffer) + '\n')


ann_dir = "/Users/rehan/workspace/data/ECB+_LREC2014"
working_folder = "../parsing/ecb"
generate_pair_sentences(ann_dir, working_folder, split='train')