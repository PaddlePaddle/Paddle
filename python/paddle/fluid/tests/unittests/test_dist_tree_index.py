# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from paddle.dataset.common import download, DATA_HOME
from paddle.distributed.fleet.dataset import TreeIndex
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle
paddle.enable_static()


def create_feeds():
    user_input = fluid.layers.data(
        name="item_id", shape=[1], dtype="int64", lod_level=1)

    item = fluid.layers.data(
        name="unit_id", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(
        name="label", shape=[1], dtype="int64", lod_level=1)
    labels = fluid.layers.data(
        name="labels", shape=[1], dtype="int64", lod_level=1)

    feed_list = [user_input, item, label, labels]
    return feed_list


class TestTreeIndex(unittest.TestCase):
    def test_tree_index(self):
        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/mini_tree.pb",
            "tree_index_unittest", "e2ba4561c2e9432b532df40546390efa")
        '''
        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/mini_tree.pb",
            "tree_index_unittest", "cadec20089f5a8a44d320e117d9f9f1a")
        '''
        tree = TreeIndex("demo", path)
        height = tree.height()
        branch = tree.branch()
        self.assertTrue(height == 5)
        self.assertTrue(branch == 2)
        self.assertEqual(tree.total_node_nums(), 25)
        self.assertEqual(tree.emb_size(), 30)

        # get_layer_codes
        layer_node_ids = []
        layer_node_codes = []
        for i in range(tree.height()):
            layer_node_codes.append(tree.get_layer_codes(i))
            layer_node_ids.append(
                [node.id() for node in tree.get_nodes(layer_node_codes[-1])])

        all_leaf_ids = [node.id() for node in tree.get_all_leafs()]
        self.assertEqual(sum(all_leaf_ids), sum(layer_node_ids[-1]))

        # get_travel
        travel_codes = tree.get_travel_codes(all_leaf_ids[0])
        travel_ids = [node.id() for node in tree.get_nodes(travel_codes)]

        for i in range(height):
            self.assertIn(travel_ids[i], layer_node_ids[height - 1 - i])
            self.assertIn(travel_codes[i], layer_node_codes[height - 1 - i])

        # get_ancestor
        ancestor_codes = tree.get_ancestor_codes([all_leaf_ids[0]], height - 2)
        ancestor_ids = [node.id() for node in tree.get_nodes(ancestor_codes)]

        self.assertEqual(ancestor_ids[0], travel_ids[1])
        self.assertEqual(ancestor_codes[0], travel_codes[1])

        # get_pi_relation
        pi_relation = tree.get_pi_relation([all_leaf_ids[0]], height - 2)
        self.assertEqual(pi_relation[all_leaf_ids[0]], ancestor_codes[0])

        # get_travel_path
        travel_path_codes = tree.get_travel_path(travel_codes[0],
                                                 travel_codes[-1])
        travel_path_ids = [
            node.id() for node in tree.get_nodes(travel_path_codes)
        ]

        self.assertEquals(travel_path_ids + [travel_ids[-1]], travel_ids)
        self.assertEquals(travel_path_codes + [travel_codes[-1]], travel_codes)

        # get_children
        children_codes = tree.get_children_codes(travel_codes[1], height - 1)
        children_ids = [node.id() for node in tree.get_nodes(children_codes)]
        self.assertIn(all_leaf_ids[0], children_ids)


class TestIndexSampler(unittest.TestCase):
    def test_layerwise_sampler(self):
        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/mini_tree.pb",
            "tree_index_unittest", "e2ba4561c2e9432b532df40546390efa")

        tdm_layer_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #tree = TreeIndex("demo", path)
        file_name = "test_in_memory_dataset_tdm_sample_run.txt"
        with open(file_name, "w") as f:
            #data = "29 d 29 d 29 29 29 29 29 29 29 29 29 29 29 29\n"
            data = "1 1 1 15 15 15\n"
            data += "1 1 1 15 15 15\n"
            f.write(data)

        slots = ["slot1", "slot2", "slot3"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(name=slot, shape=[1], dtype="int64")
            slots_vars.append(var)

        dataset = paddle.distributed.InMemoryDataset()
        dataset.init(
            batch_size=1,
            pipe_command="cat",
            download_cmd="cat",
            use_var=slots_vars)
        dataset.set_filelist([file_name])
        #dataset.update_settings(pipe_command="cat")
        #dataset._init_distributed_settings(
        #    parse_ins_id=True,
        #    parse_content=True,
        #    fea_eval=True,
        #    candidate_size=10000)

        dataset.load_into_memory()
        dataset.tdm_sample(
            'demo',
            tree_path=path,
            tdm_layer_counts=tdm_layer_counts,
            start_sample_layer=1,
            with_hierachy=False,
            seed=0,
            id_slot=2)
        self.assertTrue(dataset.get_shuffle_data_size() == 8)


if __name__ == '__main__':
    unittest.main()
