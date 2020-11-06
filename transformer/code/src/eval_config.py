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
"""Network evaluation config setting, will be used in eval.py."""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .transformer_model import TransformerConfig

cfg = edict({
    'transformer_network': 'base',
    
    'data_file': './data/test.mindrecord',
    'test_source_file':'./data/source_test.txt',
    'model_file': './ckpt/transformer-32_40-15_590.ckpt' ,
    'vocab_file':'./data1/ch_en_vocab.txt',
    'token_file': './token-32-40.txt',
    'pred_file':'./pred-32-40.txt',
    
    # -------------------mast same with the train config and the datsset------------------------
    'seq_length':40,
    'vocab_size':10067,

    #-------------------------------------eval config-----------------------------
    'batch_size':32,
    'max_position_embeddings':40       # mast same with the train config
})

'''
two kinds of transformer model version
'''
if cfg.transformer_network == 'base':
    transformer_net_cfg = TransformerConfig(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        vocab_size=cfg.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=cfg.max_position_embeddings,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        beam_width=4,
        max_decode_length=cfg.seq_length,
        length_penalty_weight=1.0,
        dtype=mstype.float32,
        compute_type=mstype.float16)
    
elif cfg.transformer_network == 'large':
    transformer_net_cfg = TransformerConfig(
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        vocab_size=cfg.vocab_size,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=cfg.max_position_embeddings,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        beam_width=4,
        max_decode_length=80,
        length_penalty_weight=1.0,
        dtype=mstype.float32,
        compute_type=mstype.float16)
else:
    raise Exception("The src/eval_confige of transformer_network must base or large and same with the train_confige confige. Change the str/eval_confige file and try again!")

