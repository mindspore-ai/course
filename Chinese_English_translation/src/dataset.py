import os
import re
import sys
import numpy as np
from mindspore import dataset as ds

def target_operation(encoder_data, decoder_data):
    encoder_data = encoder_data[1:]
    target_data = decoder_data[1:]
    decoder_data = decoder_data[:-1]
    return encoder_data, decoder_data, target_data

def eval_operation(encoder_data, decoder_data):
    encoder_data = encoder_data[1:]
    decoder_data = decoder_data[:-1]
    return encoder_data, decoder_data

def create_dataset(data_home, batch_size, repeat_num=1, is_training=True, device_num=1, rank=0):
    if is_training:
        data_dir = os.path.join(data_home, "gru_train.mindrecord")
    else:
        data_dir = os.path.join(data_home, "gru_eval.mindrecord")
    data_set = ds.MindDataset(data_dir, columns_list=["encoder_data","decoder_data"], num_parallel_workers=4,
                              num_shards=device_num, shard_id=rank)
    if is_training:
        operations = target_operation
        data_set = data_set.map(operations=operations, input_columns=["encoder_data","decoder_data"],
                    output_columns=["encoder_data","decoder_data","target_data"],
                    column_order=["encoder_data","decoder_data","target_data"])
    else:
        operations = eval_operation
        data_set = data_set.map(operations=operations, input_columns=["encoder_data","decoder_data"],
                   output_columns=["encoder_data","decoder_data"],
                   column_order=["encoder_data","decoder_data"])
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.repeat(count=repeat_num)
    return data_set
