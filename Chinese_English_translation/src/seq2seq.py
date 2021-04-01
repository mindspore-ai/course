# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Seq2Seq construction"""
import math
import numpy as np
from mindspore import Tensor
from mindspore import Tensor, Parameter
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from src.loss import NLLLoss


def gru_default_state(batch_size, input_size, hidden_size, num_layers=1, bidirectional=False):
    '''Weight init for gru cell'''
    stdv = 1 / math.sqrt(hidden_size)
    weight_i = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (input_size, 3*hidden_size)).astype(np.float32)), name='weight_i')
    weight_h = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (hidden_size, 3*hidden_size)).astype(np.float32)), name='weight_h')
    bias_i = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float32)), name='bias_i')
    bias_h = Parameter(Tensor(
        np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float32)), name='bias_h')
    return weight_i, weight_h, bias_i, bias_h

class GRU(nn.Cell):
    def __init__(self, config, is_training=True):
        super(GRU, self).__init__()
        if is_training:
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.eval_batch_size
        self.hidden_size = config.hidden_size
        self.weight_i, self.weight_h, self.bias_i, self.bias_h = \
            gru_default_state(self.batch_size, self.hidden_size, self.hidden_size)
        self.rnn = P.DynamicGRUV2()
        self.cast = P.Cast()

    def construct(self, x, hidden):
        x = self.cast(x, mstype.float16)
        y1, h1, _, _, _, _ = self.rnn(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, hidden)
        return y1, h1


class Encoder(nn.Cell):
    def __init__(self, config, is_training=True):
        super(Encoder, self).__init__()
        self.vocab_size = config.en_vocab_size
        self.hidden_size = config.hidden_size
        if is_training:
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.eval_batch_size

        self.trans = P.Transpose()
        self.perm = (1, 0, 2)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = GRU(config, is_training=is_training).to_float(mstype.float16)
        self.h = Tensor(np.zeros((self.batch_size, self.hidden_size)).astype(np.float16))

    def construct(self, encoder_input):
        embeddings = self.embedding(encoder_input)
        embeddings = self.trans(embeddings, self.perm)
        output, hidden = self.gru(embeddings, self.h)
        return output, hidden

class Decoder(nn.Cell):
    def __init__(self, config, is_training=True, dropout=0.1):
        super(Decoder, self).__init__()

        self.vocab_size = config.ch_vocab_size
        self.hidden_size = config.hidden_size
        self.max_len = config.max_seq_length

        self.trans = P.Transpose()
        self.perm = (1, 0, 2)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(1-dropout)
        self.attn = nn.Dense(self.hidden_size, self.max_len)
        self.softmax = nn.Softmax(axis=2)
        self.bmm = P.BatchMatMul()
        self.concat = P.Concat(axis=2)
        self.attn_combine = nn.Dense(self.hidden_size * 2, self.hidden_size)

        self.gru = GRU(config, is_training=is_training).to_float(mstype.float16)
        self.out = nn.Dense(self.hidden_size, self.vocab_size)
        self.logsoftmax = nn.LogSoftmax(axis=2)
        self.cast = P.Cast()

    def construct(self, decoder_input, hidden, encoder_output):
        embeddings = self.embedding(decoder_input)
        embeddings = self.dropout(embeddings)
        # calculate attn
        attn_weights = self.softmax(self.attn(embeddings))
        encoder_output = self.trans(encoder_output, self.perm)
        attn_applied = self.bmm(attn_weights, self.cast(encoder_output,mstype.float32))
        output =  self.concat((embeddings, attn_applied))
        output = self.attn_combine(output)


        embeddings = self.trans(embeddings, self.perm)
        output, hidden = self.gru(embeddings, hidden)
        output = self.cast(output, mstype.float32)
        output = self.out(output)
        output = self.logsoftmax(output)

        return output, hidden, attn_weights

class Seq2Seq(nn.Cell):
    def __init__(self, config, is_train=True):
        super(Seq2Seq, self).__init__()
        self.max_len = config.max_seq_length
        self.is_train = is_train

        self.encoder = Encoder(config, is_train)
        self.decoder = Decoder(config, is_train)
        self.expanddims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=0)
        self.argmax = P.ArgMaxWithValue(axis=int(2), keep_dims=True)
        self.concat = P.Concat(axis=1)
        self.concat2 = P.Concat(axis=0)
        self.select = P.Select()

    def construct(self, src, dst):
        encoder_output, hidden = self.encoder(src)
        decoder_hidden = self.squeeze(encoder_output[self.max_len-2:self.max_len-1:1, ::, ::])
        if self.is_train:
            outputs, _ = self.decoder(dst, decoder_hidden, encoder_output)
        else:
            decoder_input = dst[::,0:1:1]
            decoder_outputs = ()
            for i in range(0, self.max_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                decoder_hidden = self.squeeze(decoder_hidden)
                decoder_output, _ = self.argmax(decoder_output)
                decoder_output = self.squeeze(decoder_output)
                decoder_outputs += (decoder_output,)
                decoder_input = decoder_output
            outputs = self.concat(decoder_outputs)
        return outputs

class WithLossCell(nn.Cell):
    def __init__(self, backbone, config):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.batch_size = config.batch_size
        self.onehot = nn.OneHot(depth=config.ch_vocab_size)
        self._loss_fn = NLLLoss()
        self.max_len = config.max_seq_length
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.print = P.Print()

    def construct(self, src, dst, label):
        out = self._backbone(src, dst)
        loss_total = 0
        for i in range(self.batch_size):
            loss = self._loss_fn(self.squeeze(out[::,i:i+1:1,::]), self.squeeze(label[i:i+1:1, ::]))
            loss_total += loss
        loss = loss_total / self.batch_size
        return loss

class InferCell(nn.Cell):
    def __init__(self, network, config):
        super(InferCell, self).__init__(auto_prefix=False)
        self.expanddims = P.ExpandDims()
        self.network = network

    def construct(self, src, dst):
        out = self.network(src, dst)
        return out

