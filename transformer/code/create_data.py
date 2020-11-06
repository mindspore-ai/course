
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
from easydict import EasyDict as edict
import logging
import numpy as np

from mindspore.mindrecord import FileWriter
import src.tokenization as tokenization

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_sos_tokens, source_eos_tokens, target_sos_tokens, target_eos_tokens):
        self.source_sos_tokens = source_sos_tokens
        self.source_eos_tokens = source_eos_tokens
        self.target_sos_tokens = target_sos_tokens
        self.target_eos_tokens = target_eos_tokens

    def __str__(self):
        s = ""
        s += "source sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_sos_tokens]))
        s += "source eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_eos_tokens]))
        s += "target sos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_sos_tokens]))
        s += "target eos tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.target_eos_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_file(writer, instance, tokenizer, max_seq_length):
    """Create files from `SampleInstance`s."""

    def _convert_ids_and_mask(input_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    source_sos_ids, source_sos_mask = _convert_ids_and_mask(instance.source_sos_tokens)
    source_eos_ids, source_eos_mask = _convert_ids_and_mask(instance.source_eos_tokens)
    target_sos_ids, target_sos_mask = _convert_ids_and_mask(instance.target_sos_tokens)
    target_eos_ids, target_eos_mask = _convert_ids_and_mask(instance.target_eos_tokens)
    
    features = collections.OrderedDict()
    features["source_sos_ids"] = np.asarray(source_sos_ids,dtype=np.int32)
    features["source_sos_mask"] = np.asarray(source_sos_mask,dtype=np.int32)
    features["source_eos_ids"] = np.asarray(source_eos_ids,dtype=np.int32)
    features["source_eos_mask"] = np.asarray(source_eos_mask,dtype=np.int32)
    features["target_sos_ids"] = np.asarray(target_sos_ids,dtype=np.int32)
    features["target_sos_mask"] = np.asarray(target_sos_mask,dtype=np.int32)
    features["target_eos_ids"] = np.asarray(target_eos_ids,dtype=np.int32)
    features["target_eos_mask"] = np.asarray(target_eos_mask,dtype=np.int32)
    writer.write_raw_data([features])
    return features

def create_training_instance(source_words, target_words, max_seq_length):
    """Creates `SampleInstance`s for a single sentence pair."""
    EOS = "</s>"
    SOS = "<s>"

    source_sos_tokens = [SOS] + source_words
    source_eos_tokens = source_words + [EOS]
    target_sos_tokens = [SOS] + target_words
    target_eos_tokens = target_words + [EOS]

    instance = SampleInstance(
        source_sos_tokens=source_sos_tokens,
        source_eos_tokens=source_eos_tokens,
        target_sos_tokens=target_sos_tokens,
        target_eos_tokens=target_eos_tokens)
    return instance


cfg = edict({
        'input_file': './data/ch_en_all.txt',
        'vocab_file': './data/ch_en_vocab.txt',
        'train_file_mindrecord': './path_cmn/train.mindrecord',
        'eval_file_mindrecord': './path_cmn/test.mindrecord',
        'train_file_source': './path_cmn/source_train.txt',
        'eval_file_source': './path_cmn/source_test.txt',
        'num_splits':1,
        'clip_to_max_len': False,
        'max_seq_length': 40
})


def main(eval_idx):
    os.mkdir('path_cmn')
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)

    writer_train = FileWriter(cfg.train_file_mindrecord, cfg.num_splits)
    writer_eval = FileWriter(cfg.eval_file_mindrecord, cfg.num_splits)
    data_schema = {"source_sos_ids": {"type": "int32", "shape": [-1]},
                   "source_sos_mask": {"type": "int32", "shape": [-1]},
                   "source_eos_ids": {"type": "int32", "shape": [-1]},
                   "source_eos_mask": {"type": "int32", "shape": [-1]},
                   "target_sos_ids": {"type": "int32", "shape": [-1]},
                   "target_sos_mask": {"type": "int32", "shape": [-1]},
                   "target_eos_ids": {"type": "int32", "shape": [-1]},
                   "target_eos_mask": {"type": "int32", "shape": [-1]}
                   }

    writer_train.add_schema(data_schema, "tranformer train")
    writer_eval.add_schema(data_schema, "tranformer eval")

    index = 0
    f_train = open(cfg.train_file_source, 'w', encoding='utf-8')
    f_test = open(cfg.eval_file_source,'w',encoding='utf-8')
    f = open(cfg.input_file, "r", encoding='utf-8')
    for s_line in f:
        line = tokenization.convert_to_unicode(s_line)

        source_line, target_line = line.strip().split("\t")
        source_tokens = tokenizer.tokenize(source_line)
        target_tokens = tokenizer.tokenize(target_line)

        if len(source_tokens) >= (cfg.max_seq_length-1) or len(target_tokens) >= (cfg.max_seq_length-1):
            if cfg.clip_to_max_len:
                source_tokens = source_tokens[:cfg.max_seq_length-1]
                target_tokens = target_tokens[:cfg.max_seq_length-1]
            else:
                continue
        
        index = index + 1
        print(source_tokens)
        instance = create_training_instance(source_tokens, target_tokens, cfg.max_seq_length)
        
        if index in eval_idx:
            f_test.write(s_line)
            features = write_instance_to_file(writer_eval, instance, tokenizer, cfg.max_seq_length)
        else:
            f_train.write(s_line)
            features = write_instance_to_file(writer_train, instance, tokenizer, cfg.max_seq_length)
    f.close()
    f_test.close()
    f_train.close()
    writer_train.commit()
    writer_eval.commit()

if __name__ == "__main__":
    sample_num = 23607
    eval_idx = np.random.choice(sample_num, int(sample_num*0.2), replace=False)
    parser = argparse.ArgumentParser(description='Transformer creating dataset')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')   
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()
    
    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
    
    main(eval_idx)
    mox.file.copy_parallel(src_url='./path_cmn/', dst_url=args_opt.train_url)
