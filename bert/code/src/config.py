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

"""
config settings, will be used in finetune.py
"""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .bert_model import BertConfig

cfg = edict({
    'is_train': True,
    'task': 'Classification',                    # 'Classification','NER'
    'num_labels': 15,                  # 15   41
    'schema_file': r'./data/tnews/schema.json',      #  r'./data/tnews/schema.json'   r'./data/clue_ner/schema.json'    None
    'ckpt_prefix': 'bert-classification',          # 'bert-classification' 'bert-ner'  'bert-ner-crf'
    'data_file': r'./data/tnews/train.tf_record',    # r'./data/tnews/train.tf_record' r'./data/tnews/dev.tf_record'      r'./data/tnews/dev.json'
                                     # r'./data/clue_ner/train.tf_record'    r'./data/clue_ner/dev.tf_record'   r'./data/clue_ner/dev.json'
    'use_crf': False,         # only NER task is used
    'assessment_method': 'Accuracy',      # only Classification task is used   choices=["Mcc", "Spearman_correlation", "Accuracy", "F1"]

    'epoch_num': 5,
    'batch_size': 16,
    'ckpt_dir': 'model_finetune',
    'pre_training_ckpt': './ckpt/bert_base.ckpt',

    'finetune_ckpt': './ckpt/bert-classification-5_3335.ckpt',    # bert-ner-crf-5_671.ckpt  bert-ner-5_671.ckpt   bert-classification-5_3335.ckpt
    'label2id_file': './data/tnews/label2id.json',        # './data/tnews/label2id.json'   './data/clue_ner/label2id.json'
    'vocab_file': './data/vocab.txt',
    'eval_out_file': 'tnews_result.txt',      #tnews_result.txt   ner_result.txt   ner_crf_result.txt
    'optimizer': 'Lamb'
})

optimizer_cfg = edict({
    'AdamWeightDecay': edict({
        'learning_rate': 3e-5,
        'end_learning_rate': 0.0,
        'power': 5.0,
        'weight_decay': 1e-5,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
        'warmup_steps': 10000,
    }),
    'Lamb': edict({
        'learning_rate': 2e-5,
        'end_learning_rate': 0.0,
        'power': 1.0,
        'warmup_steps': 10000,
        'weight_decay': 0.01,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})


bert_net_cfg = BertConfig(
    seq_length=128,
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
