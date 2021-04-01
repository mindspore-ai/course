import os
import re
import sys
import random
import numpy as np
import unicodedata
from mindspore import dataset as ds
from mindspore.mindrecord import FileWriter

EOS = "<eos>"
SOS = "<sos>"
MAX_SEQ_LEN=10

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepare_data(data_path, vocab_save_path, max_seq_len):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # get sentences
    data = data.split('\n')

    data = data[:2000]

    # split en and chi (sentence)
    en_data = [normalizeString(line.split('\t')[0]) for line in data]
    ch_data = [line.split('\t')[1] for line in data]

    # get vocab and save
    en_vocab = set(' '.join(en_data).split(' '))
    id2en = [EOS] + [SOS] + list(en_vocab)
    en2id = {c:i for i,c in enumerate(id2en)}
    en_vocab_size = len(id2en)
    np.savetxt(os.path.join(vocab_save_path, 'en_vocab.txt'), np.array(id2en), fmt='%s')

    ch_vocab = set(''.join(ch_data))
    id2ch = [EOS] + [SOS] + list(ch_vocab)
    ch2id = {c:i for i,c in enumerate(id2ch)}
    ch_vocab_size = len(id2ch)
    np.savetxt(os.path.join(vocab_save_path, 'ch_vocab.txt'), np.array(id2ch), fmt='%s')

    # turn sentences to vocab ids --> [SOS] + sentences ids + [EOS]
    en_num_data = np.array([[1] + [int(en2id[en]) for en in line.split(' ')] + [0] for line in en_data])
    ch_num_data = np.array([[1] + [int(ch2id[ch]) for ch in line] + [0] for line in ch_data])

    #expand to max length
    for i in range(len(en_num_data)):
        num = max_seq_len + 1 - len(en_num_data[i])
        if(num >= 0):
            en_num_data[i] += [0]*num
        else:
            en_num_data[i] = en_num_data[i][:max_seq_len] + [0]

    for i in range(len(ch_num_data)):
        num = max_seq_len + 1 - len(ch_num_data[i])
        if(num >= 0):
            ch_num_data[i] += [0]*num
        else:
            ch_num_data[i] = ch_num_data[i][:max_seq_len] + [0]

    return en_num_data, ch_num_data, en_vocab_size, ch_vocab_size


def convert_to_mindrecord(data_path, mindrecord_save_path, max_seq_len):
    en_num_data, ch_num_data, en_vocab_size, ch_vocab_size = prepare_data(data_path, mindrecord_save_path, max_seq_len)

    data_list_train = []
    for en, de in zip(en_num_data, ch_num_data):
        en = np.array(en).astype(np.int32)
        de = np.array(de).astype(np.int32)
        data_json = {"encoder_data": en.reshape(-1),
                     "decoder_data": de.reshape(-1)}
        data_list_train.append(data_json)
    data_list_eval = random.sample(data_list_train, 20)

    data_dir = os.path.join(mindrecord_save_path, "gru_train.mindrecord")
    writer = FileWriter(data_dir)
    schema_json = {"encoder_data": {"type": "int32", "shape": [-1]},
                   "decoder_data": {"type": "int32", "shape": [-1]}}
    writer.add_schema(schema_json, "gru_schema")
    writer.write_raw_data(data_list_train)
    writer.commit()

    data_dir = os.path.join(mindrecord_save_path, "gru_eval.mindrecord")
    writer = FileWriter(data_dir)
    writer.add_schema(schema_json, "gru_schema")
    writer.write_raw_data(data_list_eval)
    writer.commit()

    print("en_vocab_size: ", en_vocab_size)
    print("ch_vocab_size: ", ch_vocab_size)

    return en_vocab_size, ch_vocab_size

if __name__=='__main__':
    convert_to_mindrecord("cmn_zhsim.txt", '../preprocess', MAX_SEQ_LEN)
