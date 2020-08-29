import os
import argparse
from itertools import chain
import numpy as np
from config import lstm_cfg as cfg

import mindspore.nn as nn
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.mindrecord import FileWriter
from mindspore.train import Model
from mindspore.train.callback import Callback
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
# Install gensim with 'pip install gensim'
import gensim

class ImdbParser():
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features
    """

    def __init__(self, imdb_path, glove_path, embed_size=300):
        self.__segs = ['train', 'test']
        self.__label_dic = {'pos': 1, 'neg': 0}
        self.__imdb_path = imdb_path
        self.__glove_dim = embed_size
        self.__glove_file = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')

        # properties
        self.__imdb_datas = {}
        self.__features = {}
        self.__labels = {}
        self.__vacab = {}
        self.__word2idx = {}
        self.__weight_np = {}
        self.__wvmodel = None

    def parse(self):
        """
        parse imdb data to memory
        """
        self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(self.__glove_file)

        for seg in self.__segs:
            self.__parse_imdb_datas(seg)
            self.__parse_features_and_labels(seg)
            self.__gen_weight_np(seg)

    def __parse_imdb_datas(self, seg):
        """
        load data from txt
        """
        data_lists = []
        for label_name, label_id in self.__label_dic.items():
            sentence_dir = os.path.join(self.__imdb_path, seg, label_name)
            for file in os.listdir(sentence_dir):
                with open(os.path.join(sentence_dir, file), mode='r', encoding='utf8') as f:
                    sentence = f.read().replace('\n', '')
                    data_lists.append([sentence, label_id])
        self.__imdb_datas[seg] = data_lists

    def __parse_features_and_labels(self, seg):
        """
        parse features and labels
        """
        features = []
        labels = []
        for sentence, label in self.__imdb_datas[seg]:
            features.append(sentence)
            labels.append(label)

        self.__features[seg] = features
        self.__labels[seg] = labels

        # update feature to tokenized
        self.__updata_features_to_tokenized(seg)
        # parse vacab
        self.__parse_vacab(seg)
        # encode feature
        self.__encode_features(seg)
        # padding feature
        self.__padding_features(seg)

    def __updata_features_to_tokenized(self, seg):
        tokenized_features = []
        for sentence in self.__features[seg]:
            tokenized_sentence = [word.lower() for word in sentence.split(" ")]
            tokenized_features.append(tokenized_sentence)
        self.__features[seg] = tokenized_features

    def __parse_vacab(self, seg):
        # vocab
        tokenized_features = self.__features[seg]
        vocab = set(chain(*tokenized_features))
        self.__vacab[seg] = vocab

        # word_to_idx: {'hello': 1, 'world':111, ... '<unk>': 0}
        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx[seg] = word_to_idx

    def __encode_features(self, seg):
        """ encode word to index """
        word_to_idx = self.__word2idx['train']
        encoded_features = []
        for tokenized_sentence in self.__features[seg]:
            encoded_sentence = []
            for word in tokenized_sentence:
                encoded_sentence.append(word_to_idx.get(word, 0))
            encoded_features.append(encoded_sentence)
        self.__features[seg] = encoded_features

    def __padding_features(self, seg, maxlen=200, pad=0):
        """ pad all features to the same length """
        padded_features = []
        for feature in self.__features[seg]:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while len(padded_feature) < maxlen:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        self.__features[seg] = padded_features

    def __gen_weight_np(self, seg):
        """
        generate weight by gensim
        """
        weight_np = np.zeros((len(self.__word2idx[seg]), self.__glove_dim), dtype=np.float32)
        for word, idx in self.__word2idx[seg].items():
            if word not in self.__wvmodel:
                continue
            word_vector = self.__wvmodel.get_vector(word)
            weight_np[idx, :] = word_vector

        self.__weight_np[seg] = weight_np

    def get_datas(self, seg):
        """
        return features, labels, and weight
        """
        features = np.array(self.__features[seg]).astype(np.int32)
        labels = np.array(self.__labels[seg]).astype(np.int32)
        weight = np.array(self.__weight_np[seg])
        return features, labels, weight

def _convert_to_mindrecord(data_home, features, labels, weight_np=None, training=True):
    """
    convert imdb dataset to mindrecoed dataset
    """
    if weight_np is not None:
        np.savetxt(os.path.join(data_home, 'weight.txt'), weight_np)

    # write mindrecord
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape": [-1]}}

    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord")

    def get_imdb_data(features, labels):
        data_list = []
        for i, (label, feature) in enumerate(zip(labels, features)):
            data_json = {"id": i,
                         "label": int(label),
                         "feature": feature.reshape(-1)}
            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(features, labels)
    writer.add_schema(schema_json, "nlp_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()

def convert_to_mindrecord(embed_size, aclimdb_path, preprocess_path, glove_path):
    """
    convert imdb dataset to mindrecoed dataset
    """
    parser = ImdbParser(aclimdb_path, glove_path, embed_size)
    parser.parse()

    if not os.path.exists(preprocess_path):
        print(f"preprocess path {preprocess_path} is not exist")
        os.makedirs(preprocess_path)

    train_features, train_labels, train_weight_np = parser.get_datas('train')
    _convert_to_mindrecord(preprocess_path, train_features, train_labels, train_weight_np)

    test_features, test_labels, _ = parser.get_datas('test')
    _convert_to_mindrecord(preprocess_path, test_features, test_labels, training=False)

def lstm_create_dataset(data_home, batch_size, repeat_num=1, training=True):
    """Data operations."""
    ds.config.set_seed(1)
    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord0")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord0")

    data_set = ds.MindDataset(data_dir, columns_list=["feature", "label"], num_parallel_workers=4)

    # apply map operations on images
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.repeat(count=repeat_num)

    return data_set

# Initialize short-term memory (h) and long-term memory (c) to 0
def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 1
    if bidirectional:
        num_directions = 2

    if context.get_context("device_target") == "CPU":
        h_list = []
        c_list = []
        i = 0
        while i < num_layers:
            hi = Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32))
            h_list.append(hi)
            ci = Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32))
            c_list.append(ci)
            i = i + 1
        h = tuple(h_list)
        c = tuple(c_list)
        return h, c

    h = Tensor(
        np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    c = Tensor(
        np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    return h, c

class SentimentNet(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 weight,
                 batch_size):
        super(SentimentNet, self).__init__()
        # Mapp words to vectors
        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               has_bias=True,
                               bidirectional=bidirectional,
                               dropout=0.0)

        self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)

        self.concat = P.Concat(1)
        if bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, num_classes)

    def construct(self, inputs):
        # input：(64,500,300)
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings, (self.h, self.c))
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((output[0], output[199]))
        outputs = self.decoder(encoding)
        return outputs


class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["acc"])
            print(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='true', choices=['true', 'false'],
                        help='whether to preprocess data.')
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is stored.')
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is stored.')
    parser.add_argument('--ckpt_path', type=str, default="./",
                        help='the path to save the checkpoint file.')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='the pretrained checkpoint file path.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU', 'CPU'],
                        help='the target device to run, support "GPU", "CPU". Default: "GPU".')
    args = parser.parse_args(['--device_target', 'CPU', '--preprocess', 'true'])

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.device_target)

    if args.preprocess == "true":
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)
        print("======================= Successful =======================")

    ds_train = lstm_create_dataset(args.preprocess_path, cfg.batch_size)
    ds_eval = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)

    iterator = ds_train.create_dict_iterator().get_next()
    first_batch_label = iterator["label"]
    first_batch_first_feature = iterator["feature"][0]
    print(f"The first batch contains label below:\n{first_batch_label}\n")
    print(f"The feature of the first item in the first batch is below vector:\n{first_batch_first_feature}")

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)

    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    loss_cb = LossMonitor()
    model = Model(network, loss, opt, {'acc': Accuracy()})

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    if args.device_target == "CPU":
        epoch_per_eval = {"epoch": [], "acc": []}
        eval_cb = EvalCallBack(model, ds_eval, 1, epoch_per_eval)
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb], dataset_sink_mode=False)
    else:
        epoch_per_eval = {"epoch": [], "acc": []}
        eval_cb = EvalCallBack(model, ds_eval, 1, epoch_per_eval)
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb])
    print("============== Training Success ==============")