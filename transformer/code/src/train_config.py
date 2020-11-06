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
"""Network config setting, will be used in dataset.py, train.py."""

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from .transformer_model import TransformerConfig

cfg = edict({
    #--------------------------------------nework confige-------------------------------------
    'transformer_network': 'base',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,

    'lr_schedule': edict({
        'learning_rate': 1.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
    #-----------------------------------save model confige-------------------------
    'enable_save_ckpt': True ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps':590,   #Save checkpoint steps, default is 590.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'transformer-32_40',
    'checkpoint_path':'',     #Checkpoint file path
    
    
    #-------------------------------device confige-----------------------------
    'enable_data_sink':False,   #Enable data sink, default is False.
    'device_id':0,
    'device_num':1,
    'distribute':False,
    
    # -----------------mast same with the dataset-----------------------
    'seq_length':40,
    'vocab_size':10067,
    
    #--------------------------------------------------------------------------
    'data_path':"./data/train.mindrecord",   #Data path
    'epoch_size':15,
    'batch_size':32,
    'max_position_embeddings':40,
    'enable_lossscale': False,       #Use lossscale or not, default is False.
    'do_shuffle':True       #Enable shuffle for dataset, default is True.
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
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=cfg.max_position_embeddings,
        initializer_range=0.02,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
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
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=cfg.max_position_embeddings,
        initializer_range=0.02,
        label_smoothing=0.1,
        input_mask_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float16)
else:
    raise Exception("The src/train_confige of transformer_network must base or large. Change the str/train_confige file and try again!")


