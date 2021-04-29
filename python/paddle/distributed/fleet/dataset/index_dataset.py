# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddle.fluid import core
from paddle.distributed.fleet.proto import index_dataset_pb2
import numpy as np
import struct


class Index(object):
    def __init__(self, name):
        self._name = name


class TreeIndex(Index):
    def __init__(self, name, path):
        super(TreeIndex, self).__init__(name)
        self._wrapper = core.IndexWrapper()
        self._wrapper.insert_tree_index(name, path)
        self._tree = self._wrapper.get_tree_index(name)
        self._height = self._tree.height()
        self._branch = self._tree.branch()
        self._total_node_nums = self._tree.total_node_nums()
        self._emb_size = self._tree.emb_size()
        self._layerwise_sampler = None

    def height(self):
        return self._height

    def branch(self):
        return self._branch

    def total_node_nums(self):
        return self._total_node_nums

    def emb_size(self):
        return self._emb_size

    def get_all_leafs(self):
        return self._tree.get_all_leafs()

    def get_nodes(self, codes):
        return self._tree.get_nodes(codes)

    def get_layer_codes(self, level):
        return self._tree.get_layer_codes(level)

    def get_travel_codes(self, id, start_level=0):
        return self._tree.get_travel_codes(id, start_level)

    def get_ancestor_codes(self, ids, level):
        return self._tree.get_ancestor_codes(ids, level)

    def get_children_codes(self, ancestor, level):
        return self._tree.get_children_codes(ancestor, level)

    def get_travel_path(self, child, ancestor):
        res = []
        while (child > ancestor):
            res.append(child)
            child = int((child - 1) / self._branch)
        return res

    def get_pi_relation(self, ids, level):
        codes = self.get_ancestor_codes(ids, level)
        return dict(zip(ids, codes))

    def init_layerwise_sampler(self,
                               layer_sample_counts,
                               start_sample_layer=1,
                               seed=0):
        assert self._layerwise_sampler is None
        self._layerwise_sampler = core.IndexSampler("by_layerwise", self._name)
        self._layerwise_sampler.init_layerwise_conf(layer_sample_counts,
                                                    start_sample_layer, seed)

    def layerwise_sample(self, user_input, index_input, with_hierarchy=False):
        if self._layerwise_sampler is None:
            raise ValueError("please init layerwise_sampler first.")
        return self._layerwise_sampler.sample(user_input, index_input,
                                              with_hierarchy)


class GraphIndexBuilder:
    def __init__(self, name=None, width=1, height=1, item_path_nums=1):
        self.name = name
        self.width = width  # D
        self.height = height  # K
        self.item_path_nums = item_path_nums

    def graph_init_by_random(self, input_filename, output_filename):
        total_path_num = pow(self.height, self.width)  # K^D
        item_path_dict = {}

        with open(input_filename, 'r') as f_in:
            for line in f_in:
                item_id = int(line.split()[0])
                if item_id not in item_path_dict:
                    item_path_dict[item_id] = []
                else:
                    continue

                item_path_dict[item_id] = list(
                    np.random.randint(0, total_path_num, self.item_path_nums))

        self.build(output_filename, item_path_dict)

    def build(self, output_filename, item_path_dict):
        graph_meta = index_dataset_pb2.GraphMeta()
        graph_meta.width = self.width
        graph_meta.height = self.height
        graph_meta.item_path_nums = self.item_path_nums

        with open(output_filename, 'wb') as f_out:
            kv_item = index_dataset_pb2.KVItem()
            kv_item.key = '.graph_meta'.encode("utf8")
            kv_item.value = graph_meta.SerializeToString()

            self._write_kv(f_out, kv_item.SerializeToString())

            for item in item_path_dict:
                graph_item = index_dataset_pb2.GraphItem()
                graph_item.item_id = item
                for path in item_path_dict[item]:
                    graph_item.path_id.append(path)
                node_kv_item = index_dataset_pb2.KVItem()
                node_kv_item.key = str(item).encode("utf8")
                node_kv_item.value = graph_item.SerializeToString()
                self._write_kv(f_out, node_kv_item.SerializeToString())

    def _write_kv(self, fwr, message):
        fwr.write(struct.pack('i', len(message)))
        fwr.write(message)


class GraphIndex(Index):
    def __init__(self, name, width, height, item_path_nums):
        super(GraphIndex, self).__init__(name)
        self._graph = None
        self.name = name
        self.width = width
        self.height = height
        self.item_path_nums = item_path_nums
        self._builder = GraphIndexBuilder(name, width, height, item_path_nums)
        self._wrapper = core.IndexWrapper()
        self.kd_represent_list = []
        self.gen_kd_represent(width, height)

    def _init_by_random(self):
        self._graph = core.GraphIndex()
        self._graph.initialize(self.height, self.width, self.item_path_nums)
        # self._builder.graph_init_by_random(input_filename, output_filename)
        # self._init_graph(output_filename)

    def _init_graph(self, filename):
        self._wrapper.insert_graph_index(self._name, filename)
        self._graph = self._wrapper.get_graph_index(self.name)

    def reset_graph_mapping(self):
        self._graph.reset_mapping()

    def add_graph_item_path_mapping(self, item_id, path_list):
        self._graph.add_item(item_id, path_list)

    def get_graph_item_path_dict(self):
        return self._graph.get_item_path_dict()

    def save_graph_path(self, filename):
        if self._graph != None:
            self._graph.save(filename)

    def load_graph_path(self, filename):
        if self._graph != None:
            self._graph.load(filename)

    def get_path_of_item(self, id):
        if isinstance(id, list):
            assert len(id) > 0
            # assert isinstance(id[0], int)
            return self._graph.get_path_of_item(id)
        elif isinstance(id, int):
            return self._graph.get_path_of_item([id])
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))

    def get_item_of_path(self, path):
        if isinstance(path, list):
            assert len(path) > 0
            # assert isinstance(path[0], int)
            return self._graph.get_item_of_path(path)
        elif isinstance(path, int):
            return self._graph.get_item_of_path([path])
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))

    def gen_kd_represent(self, width, height):
        def recursive_method(start_idx, cur_path):
            if start_idx == width:
                self.kd_represent_list.append(list(cur_path))
                return

            for i in range(height):
                cur_path.append(i)
                recursive_method(start_idx + 1, cur_path)
                cur_path.pop()

        init_idx = 0
        init_path = []
        recursive_method(init_idx, init_path)
        return

    def path_id_to_kd_represent(self, path_id):
        assert 0 <= path_id and path_id < pow(self.height, self.width)
        return self.kd_represent_list[path_id]

    def kd_represent_to_path_id(self, kd_represent):
        assert len(kd_represent) == self.width
        path_id = 0
        for idx, val in enumerate(reversed(kd_represent)):
            assert 0 <= val and val < self.height
            path_id += val * pow(self.height, idx)
        return path_id

    def update_path_of_item(self, item_paths):
        if isinstance(item_paths, dict):
            assert len(item_paths) > 0
            assert isinstance(item_paths[0], list)
            return self._graph.update_path_of_item(item_paths)
        elif isinstance(item_paths, int):  # {int, ["",""]}
            return self._graph.update_path_of_item({item_paths, []})
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))

#   int update_Jpath_of_item(
#     std::map<uint64_t, std::vector<std::string>>& item_paths, const int T, const int J, const double lambda, const int factor);
#  J=self.item_path_nums

    def update_Jpath_of_item(self, item_paths, T=3, J=3, lamd=1e-7, factor=2):

        if isinstance(item_paths, dict):
            assert len(item_paths) > 0
            assert isinstance(item_paths[0], list)
            return self._graph.update_Jpath_of_item(item_paths, T, J, lamd,
                                                    factor)
        elif isinstance(item_paths, int):  # {int, ["",""]}
            return self._graph.update_Jpath_of_item({item_paths, []}, T, J,
                                                    lamd, factor)
        else:
            raise ValueError(
                "Illegal input type {}, required list or int".format(type(id)))
