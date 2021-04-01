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
from paddle.distributed.fleet.data_generator import TreeIndex
from paddle.fluid.core import IndexSampler


class TestTreeIndex(unittest.TestCase):
    def test_tree_index(self):
        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/demo_tree.pb",
            "tree_index_unittest", "cadec20089f5a8a44d320e117d9f9f1a")

        tree = TreeIndex("demo", path)
        height = tree.height()
        branch = tree.branch()
        self.assertTrue(height == 14)
        self.assertTrue(branch == 2)
        self.assertEqual(tree.total_node_nums(), 15581)
        self.assertEqual(tree.tree_max_node(), 5171135)

        # get_nodes_given_level
        layer_node_ids = []
        layer_node_codes = []
        for i in range(tree.height()):
            layer_node_ids.append(tree.get_nodes_given_level(i, False))
            layer_node_codes.append(tree.get_nodes_given_level(i, True))
            self.assertTrue(
                len(layer_node_ids[-1]) == len(layer_node_codes[-1]))

        all_items = tree.get_all_items()
        self.assertEqual(sum(all_items), sum(layer_node_ids[-1]))

        # get_parent_path
        parent_path_ids = tree.get_parent_path([all_items[0]])[0]
        parent_path_codes = tree.get_parent_path(
            [all_items[0]], ret_code=True)[0]
        for i in range(height):
            self.assertIn(parent_path_ids[i], layer_node_ids[height - 1 - i])
            self.assertIn(parent_path_codes[i],
                          layer_node_codes[height - 1 - i])

        # get_ancestor_given_level
        ancestor_ids = tree.get_ancestor_given_level([all_items[0]], height - 2,
                                                     False)
        ancestor_codes = tree.get_ancestor_given_level([all_items[0]],
                                                       height - 2, True)
        self.assertEqual(ancestor_ids[0], parent_path_ids[1])
        self.assertEqual(ancestor_codes[0], parent_path_codes[1])

        # get_pi_relation
        pi_relation = tree.get_pi_relation([all_items[0]], height - 2)
        self.assertEqual(pi_relation[all_items[0]], ancestor_codes[0])

        # get_travel_path
        travel_path_ids = tree.get_travel_path(parent_path_codes[0],
                                               parent_path_codes[-1])
        self.assertEquals(travel_path_ids + [parent_path_ids[-1]],
                          parent_path_ids)
        travel_path_codes = tree.get_travel_path(parent_path_codes[0],
                                                 parent_path_codes[-1], True)
        self.assertEquals(travel_path_codes + [parent_path_codes[-1]],
                          parent_path_codes)

        children = tree.get_children_given_ancestor_and_level(
            parent_path_codes[1], height - 1, False)
        self.assertIn(all_items[0], children)


class TestIndexSampler(unittest.TestCase):
    def test_layerwise_sampler(self):
        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/demo_tree.pb",
            "tree_index_unittest", "cadec20089f5a8a44d320e117d9f9f1a")

        tree = TreeIndex("demo", path)
        sampler = IndexSampler("by_layerwise", "demo")

        layer_nodes = []
        for i in range(tree.height()):
            layer_nodes.append(tree.get_nodes_given_level(i, False))

        sample_num = range(1, 10000)
        start_sample_layer = 1
        seed = 0
        sample_layers = tree.height() - start_sample_layer
        sample_num = sample_num[:sample_layers]
        layer_sample_counts = sample_num + [1] * (sample_layers -
                                                  len(sample_num))
        total_sample_num = sum(layer_sample_counts) + len(layer_sample_counts)
        sampler.init_layerwise_conf(sample_num, start_sample_layer, seed)

        ids = [315757, 838060, 1251533, 403522, 2473624, 3321007]
        tmp = tree.get_parent_path(ids, start_sample_layer, False)
        parent_path = {}
        for i in range(len(ids)):
            parent_path[ids[i]] = tmp[i]
        # print(parent_path)

        # 2. check sample res with_hierarchy = False
        sample_res = sampler.sample([[315757, 838060], [1251533, 403522]],
                                    [2473624, 3321007], False)
        # print(sample_res)
        idx = 0
        layer = tree.height() - 1
        for i in range(len(layer_sample_counts)):
            for j in range(layer_sample_counts[0 - (i + 1)] + 1):
                self.assertTrue(sample_res[idx + j][0] == 315757)
                self.assertTrue(sample_res[idx + j][1] == 838060)
                self.assertTrue(sample_res[idx + j][2] in layer_nodes[layer])
                if j == 0:
                    self.assertTrue(sample_res[idx + j][3] == 1)
                    self.assertTrue(
                        sample_res[idx + j][2] == parent_path[2473624][i])
                else:
                    self.assertTrue(sample_res[idx + j][3] == 0)
                    self.assertTrue(
                        sample_res[idx + j][2] != parent_path[2473624][i])
            idx += layer_sample_counts[0 - (i + 1)] + 1
            layer -= 1
        self.assertTrue(idx == total_sample_num)
        layer = tree.height() - 1
        for i in range(len(layer_sample_counts)):
            for j in range(layer_sample_counts[0 - (i + 1)] + 1):
                self.assertTrue(sample_res[idx + j][0] == 1251533)
                self.assertTrue(sample_res[idx + j][1] == 403522)
                self.assertTrue(sample_res[idx + j][2] in layer_nodes[layer])
                if j == 0:
                    self.assertTrue(sample_res[idx + j][3] == 1)
                    self.assertTrue(
                        sample_res[idx + j][2] == parent_path[3321007][i])
                else:
                    self.assertTrue(sample_res[idx + j][3] == 0)
                    self.assertTrue(
                        sample_res[idx + j][2] != parent_path[3321007][i])
            idx += layer_sample_counts[0 - (i + 1)] + 1
            layer -= 1
        self.assertTrue(idx == total_sample_num * 2)

        # 3. check sample res with_hierarchy = True
        sample_res_with_hierarchy = sampler.sample(
            [[315757, 838060], [1251533, 403522]], [2473624, 3321007], True)
        # print(sample_res_with_hierarchy)
        idx = 0
        layer = tree.height() - 1
        for i in range(len(layer_sample_counts)):
            for j in range(layer_sample_counts[0 - (i + 1)] + 1):
                self.assertTrue(sample_res_with_hierarchy[idx + j][0] ==
                                parent_path[315757][i])
                self.assertTrue(sample_res_with_hierarchy[idx + j][1] ==
                                parent_path[838060][i])
                self.assertTrue(
                    sample_res_with_hierarchy[idx + j][2] in layer_nodes[layer])
                if j == 0:
                    self.assertTrue(sample_res_with_hierarchy[idx + j][3] == 1)
                    self.assertTrue(sample_res_with_hierarchy[idx + j][2] ==
                                    parent_path[2473624][i])
                else:
                    self.assertTrue(sample_res_with_hierarchy[idx + j][3] == 0)
                    self.assertTrue(sample_res_with_hierarchy[idx + j][2] !=
                                    parent_path[2473624][i])

            idx += layer_sample_counts[0 - (i + 1)] + 1
            layer -= 1
        self.assertTrue(idx == total_sample_num)
        layer = tree.height() - 1
        for i in range(len(layer_sample_counts)):
            for j in range(layer_sample_counts[0 - (i + 1)] + 1):
                self.assertTrue(sample_res_with_hierarchy[idx + j][0] ==
                                parent_path[1251533][i])
                self.assertTrue(sample_res_with_hierarchy[idx + j][1] ==
                                parent_path[403522][i])
                self.assertTrue(
                    sample_res_with_hierarchy[idx + j][2] in layer_nodes[layer])
                if j == 0:
                    self.assertTrue(sample_res_with_hierarchy[idx + j][3] == 1)
                    self.assertTrue(sample_res_with_hierarchy[idx + j][2] ==
                                    parent_path[3321007][i])
                else:
                    self.assertTrue(sample_res_with_hierarchy[idx + j][3] == 0)
                    self.assertTrue(sample_res_with_hierarchy[idx + j][2] !=
                                    parent_path[3321007][i])

            idx += layer_sample_counts[0 - (i + 1)] + 1
            layer -= 1
        self.assertTrue(idx == 2 * total_sample_num)


if __name__ == '__main__':
    unittest.main()
