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


'''bert clue evaluation'''

import json
import re

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from src import tokenization

from src.CRF import postprocess
from src.config import bert_net_cfg, cfg
from src.tokenization import convert_tokens_to_ids

"""process txt"""


def process_one_example_p(tokenizer, vocab, text, max_seq_len=128):
    """process one testline"""
    textlist = list(text)
    tokens = []
    for _, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for _, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = convert_tokens_to_ids(vocab, ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def label_generation(text="", probs=None, tag_to_index=None):
    """generate label"""
    data = [text]
    probs = [probs]
    result = []
    label2id = tag_to_index
    id2label = [k for k, v in label2id.items()]

    for index, prob in enumerate(probs):
        for v in prob[1:len(data[index]) + 1]:
            result.append(id2label[int(v)])

    labels = {}
    start = None
    index = 0
    for _, t in zip("".join(data), result):
        if re.search("^[BS]", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    labels[label] = {te_: [[start, index - 1]]}
            start = index
        if re.search("^O", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    labels[label] = {te_: [[start, index - 1]]}
            start = None
        index += 1
    if start is not None:
        label = result[start][2:]
        if labels.get(label):
            te_ = text[start:index]
            labels[label][te_] = [[start, index - 1]]
        else:
            te_ = text[start:index]
            labels[label] = {te_: [[start, index - 1]]}
    return labels


f = open(cfg.eval_out_file, 'w')
def process(model=None, text="", tokenizer_=None, use_crf=False, tag_to_index=None, vocab=""):
    """
    process text.
    """
    data = [text]
    features = []
    res = []
    ids = []
    for i in data:
        f.write("text: " + str(i) + '\n')
        feature = process_one_example_p(tokenizer_, vocab, i, max_seq_len=bert_net_cfg.seq_length)
        features.append(feature)
        input_ids, input_mask, token_type_id = feature
        f.write("input_ids:  " + str(input_ids) + '\n')
        f.write("input_mask:  " + str(input_mask) + '\n')
        f.write("segment_ids: " + str(token_type_id) + '\n')
        input_ids = Tensor(np.array(input_ids), mstype.int32)
        input_mask = Tensor(np.array(input_mask), mstype.int32)
        token_type_id = Tensor(np.array(token_type_id), mstype.int32)
        if use_crf:
            backpointers, best_tag_id = model.predict(input_ids, input_mask, token_type_id, Tensor(1))
            best_path = postprocess(backpointers, best_tag_id)
            logits = []
            for ele in best_path:
                logits.extend(ele)
            ids = logits
        else:
            logits = model.predict(input_ids, input_mask, token_type_id, Tensor(1))
            ids = logits.asnumpy()
            ids = np.argmax(ids, axis=-1)
            ids = list(ids)
            f.write("pre_labels: " + str(ids) + '\n')
    res = label_generation(text=text, probs=ids, tag_to_index=tag_to_index)
    return res


def submit(model=None, path="", vocab_file="", use_crf="", label_file="", tag_to_index=None):
    """
    submit task
    """
    tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
    data = []
    if cfg.schema_file is not None:
        f1 = open(cfg.schema_file, 'r')
        numRows = json.load(f1)
        up_num = numRows["numRows"]
    else:
        up_num = 600000000000
    num = 0
    for line in open(path):
        num = num + 1
        if num > up_num:
            break
        if not line.strip():
            continue
        oneline = json.loads(line.strip())
        if cfg.task == 'Classification':
            res = process(model=model, text=oneline["sentence"], tokenizer_=tokenizer_,
                          use_crf=use_crf, tag_to_index=tag_to_index, vocab=vocab_file)

            print("text", oneline["sentence"])
        elif cfg.task == 'NER':
            res = process(model=model, text=oneline["text"], tokenizer_=tokenizer_,
                          use_crf=use_crf, tag_to_index=tag_to_index, vocab=vocab_file)
            print("text", oneline["text"])
        else:
            raise Exception("Task error")
        print("res:", res)
        f.write("result: " + str(res) + '\n')
        data.append(json.dumps({"label": res}, ensure_ascii=False))
    f.close()