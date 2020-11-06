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
"""Transformer evaluation script."""
import argparse

import os
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import context
import src.tokenization as tokenization

from src.transformer_model import TransformerModel
from src.eval_config import cfg, transformer_net_cfg

def load_test_data(batch_size=1, data_file=None):
    """
    Load test dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["source_eos_ids", "source_eos_mask",
                                      "target_sos_ids", "target_sos_mask",
                                      "target_eos_ids", "target_eos_mask"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="source_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_eos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_mask", operations=type_cast_op)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'transformer'
    return ds

class TransformerInferCell(nn.Cell):
    """
    Encapsulation class of transformer network infer.
    """
    def __init__(self, network):
        super(TransformerInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids,
                  source_mask):
        predicted_ids = self.network(source_ids, source_mask)
        return predicted_ids

def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = load_checkpoint(model_path)
        is_npz = False

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        if "tfm_decoder" in msname:
            infer_name = "tfm_decoder.decoder." + infer_name
        if is_npz:
            weights[infer_name] = ms_ckpt[msname]
        else:
            weights[infer_name] = ms_ckpt[msname].data.asnumpy()
    weights["tfm_decoder.decoder.tfm_embedding_lookup.embedding_table"] = \
        weights["tfm_embedding_lookup.embedding_table"]

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict

def run_transformer_eval(out_url):
    """
    Transformer evaluation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", reserve_class_name_in_scope=False)

    tfm_model = TransformerModel(config=transformer_net_cfg, is_training=False, use_one_hot_embeddings=False)
    print(cfg.model_file)
    parameter_dict = load_weights(cfg.model_file)
    load_param_into_net(tfm_model, parameter_dict)
    tfm_infer = TransformerInferCell(tfm_model)
    model = Model(tfm_infer)
    
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)
    dataset = load_test_data(batch_size=cfg.batch_size, data_file=cfg.data_file)
    predictions = []
    source_sents = []
    target_sents = []
    f = open(cfg.token_file, 'w', encoding='utf-8')
    f1 = open(cfg.pred_file, 'w', encoding='utf-8')
    f2 = open(cfg.test_source_file, 'r', encoding='utf-8')
    for batch in dataset.create_dict_iterator():
        source_sents.append(batch["source_eos_ids"])
        target_sents.append(batch["target_eos_ids"])
        source_ids = Tensor(batch["source_eos_ids"], mstype.int32)
        source_mask = Tensor(batch["source_eos_mask"], mstype.int32)
        predicted_ids = model.predict(source_ids, source_mask)
        #predictions.append(predicted_ids.asnumpy())
        # ----------------------------------------decode and write to file(token file)---------------------
        batch_out = predicted_ids.asnumpy()
        for i in range(transformer_net_cfg.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            token_ids = [str(x) for x in batch_out[i].tolist()]
            print(" ".join(token_ids))
            token=" ".join(token_ids)
            f.write(token + "\n")
            #-------------------------------token_ids to real output file-------------------------------
            token_ids = [int(x) for x in token.strip().split()]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sent = " ".join(tokens)
            sent = sent.split("<s>")[-1]
            sent = sent.split("</s>")[0]
            print(sent.strip())
            f1.write(f2.readline().strip()+'\t')
            f1.write(sent.strip()+'\n')
    f.close()
    f1.close()
    f2.close()

    import moxing as mox
    mox.file.copy_parallel(src_url=cfg.token_file, dst_url=os.path.join(out_url,cfg.token_file))
    mox.file.copy_parallel(src_url=cfg.pred_file, dst_url=os.path.join(out_url,cfg.pred_file))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer testing')
    parser.add_argument('--data_url', required=True, default=None, help='Location of pre data.')   
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--ckpt_url', required=True, default=None, help='Location of model.') 
    parser.add_argument('--data_source', required=True, default=None, help='Location of source data.') 
    args_opt = parser.parse_args()
    
    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
    mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url='./ckpt/')
    mox.file.copy_parallel(src_url=args_opt.data_source, dst_url='./data1/')
    
    run_transformer_eval(args_opt.train_url)
