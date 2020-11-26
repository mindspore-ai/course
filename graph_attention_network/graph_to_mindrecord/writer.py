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
# ==============================================================================
"""
######################## write mindrecord example ########################
Write mindrecord by data dictionary:
python writer.py --mindrecord_script /YourScriptPath ...
"""
import os
import time
import shutil

from multiprocessing import Pool
from importlib import import_module
from easydict import EasyDict as edict
from mindspore.mindrecord import FileWriter
from .graph_map_schema import GraphMapSchema
# import cora.mr_api

def read_args():
    """
    read args
    """
    ret_args = edict({
    'mindrecord_script': 'template',
    'mindrecord_file': '/tmp/mindrecord/xyz',
    'mindrecord_partitions': 1,
    'mindrecord_header_size_by_bit':24,
    'mindrecord_page_size_by_bit' : 25,
    'mindrecord_workers' : 8,
    'num_node_tasks': 1,
    'num_edge_tasks' : 1,
    'graph_api_args':'/tmp/nodes.csv:/tmp/edges.csv',
})

    return ret_args


def run(cfg):
    args = read_args()
    #建立输出文件夹
    cur_path = os.getcwd()
    M_PATH = os.path.join(cur_path, cfg.MINDRECORD_PATH)
    if os.path.exists(M_PATH):
        shutil.rmtree(M_PATH)  # 删除文件夹
    os.mkdir(M_PATH)
    cfg.SRC_PATH = os.path.join(cur_path, cfg.SRC_PATH)
    #参数
    args.mindrecord_script= cfg.DATASET_NAME
    args.mindrecord_file=os.path.join(cfg.MINDRECORD_PATH,cfg.DATASET_NAME)
    args.mindrecord_partitions=cfg.mindrecord_partitions
    args.mindrecord_header_size_by_bit=cfg.mindrecord_header_size_by_bit
    args.mindrecord_page_size_by_bit=cfg.mindrecord_header_size_by_bit
    args.graph_api_args=cfg.SRC_PATH

    start_time = time.time()
    # pass mr_api arguments
    os.environ['graph_api_args'] = args.graph_api_args

    try:
        mr_api = import_module('graph_to_mindrecord.'+args.mindrecord_script + '.mr_api')
    except ModuleNotFoundError:
        raise RuntimeError("Unknown module path: {}".format(args.mindrecord_script + '.mr_api'))

    # init graph schema
    graph_map_schema = GraphMapSchema()

    num_features, feature_data_types, feature_shapes = mr_api.node_profile
    graph_map_schema.set_node_feature_profile(num_features, feature_data_types, feature_shapes)

    num_features, feature_data_types, feature_shapes = mr_api.edge_profile
    graph_map_schema.set_edge_feature_profile(num_features, feature_data_types, feature_shapes)

    graph_schema = graph_map_schema.get_schema()
    #----------------------------------------------------------------------------------------------
    def init_writer(mr_schema):
        """
        init writer
        """
        print("Init writer  ...")
        mr_writer = FileWriter(args.mindrecord_file, args.mindrecord_partitions)

        # set the header size
        if args.mindrecord_header_size_by_bit != 24:
            header_size = 1 << args.mindrecord_header_size_by_bit
            mr_writer.set_header_size(header_size)

        # set the page size
        if args.mindrecord_page_size_by_bit != 25:
            page_size = 1 << args.mindrecord_page_size_by_bit
            mr_writer.set_page_size(page_size)

        # create the schema
        mr_writer.add_schema(mr_schema, "mindrecord_graph_schema")

        # open file and set header
        mr_writer.open_and_set_header()

        return mr_writer

    def run_parallel_workers():
        """
        run parallel workers
        """
        # set number of workers
        num_workers = args.mindrecord_workers

        task_list = list(range(args.num_node_tasks))

        if num_workers > args.num_node_tasks:
            num_workers = args.num_node_tasks

        if os.name == 'nt':
            for window_task_id in task_list:
                exec_task(window_task_id, False)
        elif args.num_node_tasks > 1:
            with Pool(num_workers) as p:
                p.map(exec_task, task_list)
        else:
            exec_task(0, False)

    def exec_task(task_id, parallel_writer=True):
        """
        Execute task with specified task id
        """
        print("exec task {}, parallel: {} ...".format(task_id, parallel_writer))
        imagenet_iter = mindrecord_dict_data(task_id)
        batch_size = 512
        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data = imagenet_iter.__next__()
                    if 'dst_id' in data:
                        data = graph_map_schema.transform_edge(data)
                    else:
                        data = graph_map_schema.transform_node(data)
                    data_list.append(data)
                    transform_count += 1
                writer.write_raw_data(data_list, parallel_writer=parallel_writer)
                print("transformed {} record...".format(transform_count))
            except StopIteration:
                if data_list:
                    writer.write_raw_data(data_list, parallel_writer=parallel_writer)
                    print("transformed {} record...".format(transform_count))
                break
    #-------------------------------------------------------------------------------------
    # init writer
    writer = init_writer(graph_schema)

    # write nodes data
    mindrecord_dict_data = mr_api.yield_nodes
    run_parallel_workers()

    # write edges data
    mindrecord_dict_data = mr_api.yield_edges
    run_parallel_workers()

    # writer wrap up
    ret = writer.commit()

    end_time = time.time()
    print("--------------------------------------------")
    print("END. Total time: {}".format(end_time - start_time))
    print("--------------------------------------------")
