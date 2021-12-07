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
        self.assertEqual(tree.emb_size(), 5171136)

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
            "https://paddlerec.bj.bcebos.com/tree-based/data/demo_tree.pb",
            "tree_index_unittest", "cadec20089f5a8a44d320e117d9f9f1a")

        tree = TreeIndex("demo", path)

        layer_nodes = []
        for i in range(tree.height()):
            layer_codes = tree.get_layer_codes(i)
            layer_nodes.append(
                [node.id() for node in tree.get_nodes(layer_codes)])

        sample_num = range(1, 10000)
        start_sample_layer = 1
        seed = 0
        sample_layers = tree.height() - start_sample_layer
        sample_num = sample_num[:sample_layers]
        layer_sample_counts = list(sample_num) + [1] * (sample_layers -
                                                        len(sample_num))
        total_sample_num = sum(layer_sample_counts) + len(layer_sample_counts)
        tree.init_layerwise_sampler(sample_num, start_sample_layer, seed)

        ids = [315757, 838060, 1251533, 403522, 2473624, 3321007]
        parent_path = {}
        for i in range(len(ids)):
            tmp = tree.get_travel_codes(ids[i], start_sample_layer)
            parent_path[ids[i]] = [node.id() for node in tree.get_nodes(tmp)]

        # check sample res with_hierarchy = False
        sample_res = tree.layerwise_sample(
            [[315757, 838060], [1251533, 403522]], [2473624, 3321007], False)
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

        # check sample res with_hierarchy = True
        sample_res_with_hierarchy = tree.layerwise_sample(
            [[315757, 838060], [1251533, 403522]], [2473624, 3321007], True)
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
