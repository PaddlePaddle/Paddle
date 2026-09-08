# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.distributed.fleet.dataset.index_dataset
# 自动生成的单测，覆盖 index_dataset 模块中未覆盖的代码
# Target: cover uncovered lines 26,31-39,42,45,48,51,54,57,60,63,66,69,72-76,79-80,88-92,100-104
#   in python/paddle/distributed/fleet/dataset/index_dataset.py
# 未覆盖行: Index._name, TreeIndex.__init__ 中的属性赋值, 各种访问方法, get_travel_path,
#           get_pi_relation, init_layerwise_sampler, layerwise_sample

import unittest
from unittest.mock import MagicMock, patch

# Import the module object directly for patching core
# 直接导入模块对象用于 patching core
import paddle.distributed.fleet.dataset.index_dataset as idx_mod
from paddle.distributed.fleet.dataset.index_dataset import (
    Index,
    TreeIndex,
)


class TestIndex(unittest.TestCase):
    """Test the Index base class.
    测试 Index 基类。"""

    def test_index_init(self):
        """Index.__init__ should set _name attribute.
        Index.__init__ 应设置 _name 属性。"""
        idx = Index("test_name")
        self.assertEqual(idx._name, "test_name")

    def test_index_name_attribute(self):
        """Index stores name as _name.
        Index 将 name 存储为 _name。"""
        idx = Index("my_index")
        self.assertEqual(idx._name, "my_index")


class TestTreeIndex(unittest.TestCase):
    """Test the TreeIndex class with mocked core.IndexWrapper.
    使用模拟的 core.IndexWrapper 测试 TreeIndex 类。"""

    def setUp(self):
        """Set up mocks for core.IndexWrapper.
        设置 core.IndexWrapper 的模拟。"""
        # Create mock tree object
        # 创建模拟 tree 对象
        self.mock_tree = MagicMock()
        self.mock_tree.height.return_value = 3
        self.mock_tree.branch.return_value = 2
        self.mock_tree.total_node_nums.return_value = 14
        self.mock_tree.emb_size.return_value = 128
        self.mock_tree.get_all_leaves.return_value = [
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        self.mock_tree.get_nodes.return_value = ["node_a", "node_b"]
        self.mock_tree.get_layer_codes.return_value = [1, 2, 3, 4]
        self.mock_tree.get_travel_codes.return_value = [1, 3, 7]
        self.mock_tree.get_ancestor_codes.return_value = [0, 0, 1]
        self.mock_tree.get_children_codes.return_value = [4, 5]

        # Create mock wrapper
        # 创建模拟 wrapper
        self.mock_wrapper = MagicMock()
        self.mock_wrapper.get_tree_index.return_value = self.mock_tree

    def test_tree_index_init(self):
        """TreeIndex.__init__ should call insert_tree_index and set attributes.
        TreeIndex.__init__ 应调用 insert_tree_index 并设置属性。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper

            idx = TreeIndex("test_tree", "/path/to/tree")

            # Verify core calls
            # 验证 core 调用
            mock_core.IndexWrapper.assert_called_once()
            self.mock_wrapper.insert_tree_index.assert_called_once_with(
                "test_tree", "/path/to/tree"
            )
            self.mock_wrapper.get_tree_index.assert_called_once_with(
                "test_tree"
            )

        # Verify attributes are set (lines 35-39)
        # 验证属性已设置（35-39行）
        self.assertEqual(idx._height, 3)
        self.assertEqual(idx._branch, 2)
        self.assertEqual(idx._total_node_nums, 14)
        self.assertEqual(idx._emb_size, 128)
        self.assertIsNone(idx._layerwise_sampler)

    def test_height(self):
        """TreeIndex.height() returns stored height.
        TreeIndex.height() 返回存储的高度。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        self.assertEqual(idx.height(), 3)

    def test_branch(self):
        """TreeIndex.branch() returns stored branch.
        TreeIndex.branch() 返回存储的分支数。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        self.assertEqual(idx.branch(), 2)

    def test_total_node_nums(self):
        """TreeIndex.total_node_nums() returns stored total node count.
        TreeIndex.total_node_nums() 返回存储的节点总数。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        self.assertEqual(idx.total_node_nums(), 14)

    def test_emb_size(self):
        """TreeIndex.emb_size() returns stored embedding size.
        TreeIndex.emb_size() 返回存储的嵌入维度。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        self.assertEqual(idx.emb_size(), 128)

    def test_get_all_leaves(self):
        """TreeIndex.get_all_leaves() delegates to tree object.
        TreeIndex.get_all_leaves() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        leaves = idx.get_all_leaves()
        self.assertEqual(leaves, [8, 9, 10, 11, 12, 13, 14, 15])

    def test_get_nodes(self):
        """TreeIndex.get_nodes() delegates to tree object.
        TreeIndex.get_nodes() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        nodes = idx.get_nodes([1, 2, 3])
        self.assertEqual(nodes, ["node_a", "node_b"])

    def test_get_layer_codes(self):
        """TreeIndex.get_layer_codes() delegates to tree object.
        TreeIndex.get_layer_codes() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        codes = idx.get_layer_codes(1)
        self.assertEqual(codes, [1, 2, 3, 4])

    def test_get_travel_codes(self):
        """TreeIndex.get_travel_codes() delegates to tree object.
        TreeIndex.get_travel_codes() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        codes = idx.get_travel_codes(7, start_level=0)
        self.assertEqual(codes, [1, 3, 7])

    def test_get_ancestor_codes(self):
        """TreeIndex.get_ancestor_codes() delegates to tree object.
        TreeIndex.get_ancestor_codes() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        codes = idx.get_ancestor_codes([7, 8, 9], level=1)
        self.assertEqual(codes, [0, 0, 1])

    def test_get_children_codes(self):
        """TreeIndex.get_children_codes() delegates to tree object.
        TreeIndex.get_children_codes() 委托给 tree 对象。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")
        codes = idx.get_children_codes(1, level=2)
        self.assertEqual(codes, [4, 5])

    def test_get_travel_path(self):
        """TreeIndex.get_travel_path() computes path from child to ancestor.
        TreeIndex.get_travel_path() 计算从子节点到祖先节点的路径。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")

        # With branch=2, path from node 7 to ancestor 1:
        # 7 -> (7-1)/2=3 -> (3-1)/2=1, so path = [7, 3]
        # branch=2 时，从节点7到祖先1的路径：
        # 7 -> (7-1)/2=3 -> (3-1)/2=1, 路径 = [7, 3]
        path = idx.get_travel_path(child=7, ancestor=1)
        self.assertEqual(path, [7, 3])

    def test_get_travel_path_same_node(self):
        """get_travel_path returns empty when child equals ancestor.
        当子节点等于祖先节点时，get_travel_path 返回空列表。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")

        path = idx.get_travel_path(child=5, ancestor=5)
        self.assertEqual(path, [])

    def test_get_travel_path_long_path(self):
        """get_travel_path with a longer path.
        较长路径的 get_travel_path。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")

        # branch=2: 15 -> 7 -> 3 -> 1
        path = idx.get_travel_path(child=15, ancestor=1)
        self.assertEqual(path, [15, 7, 3])

    def test_get_pi_relation(self):
        """TreeIndex.get_pi_relation() maps ids to ancestor codes.
        TreeIndex.get_pi_relation() 将 id 映射到祖先编码。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")

        relation = idx.get_pi_relation([7, 8, 9], level=1)
        # Should return dict(zip([7,8,9], [0,0,1]))
        # 应返回 dict(zip([7,8,9], [0,0,1]))
        self.assertEqual(relation, {7: 0, 8: 0, 9: 1})

    def test_init_layerwise_sampler(self):
        """TreeIndex.init_layerwise_sampler() creates sampler.
        TreeIndex.init_layerwise_sampler() 创建采样器。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            mock_sampler = MagicMock()
            mock_core.IndexSampler.return_value = mock_sampler

            idx = TreeIndex("test_tree", "/path")
            idx.init_layerwise_sampler(
                layer_sample_counts=[10, 5, 2],
                start_sample_layer=1,
                seed=42,
            )

            mock_core.IndexSampler.assert_called_once_with(
                "by_layerwise", "test_tree"
            )
            mock_sampler.init_layerwise_conf.assert_called_once_with(
                [10, 5, 2], 1, 42
            )

    def test_init_layerwise_sampler_already_initialized(self):
        """init_layerwise_sampler raises if called twice.
        重复调用 init_layerwise_sampler 会引发错误。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            mock_sampler = MagicMock()
            mock_core.IndexSampler.return_value = mock_sampler

            idx = TreeIndex("test_tree", "/path")
            idx.init_layerwise_sampler([10, 5, 2])

            # Second call should raise AssertionError
            # 第二次调用应引发 AssertionError
            with self.assertRaises(AssertionError):
                idx.init_layerwise_sampler([10, 5, 2])

    def test_layerwise_sample_without_init(self):
        """layerwise_sample raises ValueError if sampler not initialized.
        如果采样器未初始化，layerwise_sample 引发 ValueError。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            idx = TreeIndex("test_tree", "/path")

        with self.assertRaises(ValueError):
            idx.layerwise_sample([[1, 2]], [0, 1])

    def test_layerwise_sample_with_init(self):
        """layerwise_sample delegates to sampler after init.
        初始化后 layerwise_sample 委托给采样器。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            mock_sampler = MagicMock()
            mock_sampler.sample.return_value = [[1, 2], [3, 4]]
            mock_core.IndexSampler.return_value = mock_sampler

            idx = TreeIndex("test_tree", "/path")
            idx.init_layerwise_sampler([10, 5])
            result = idx.layerwise_sample([[1, 2]], [0, 1])

            self.assertEqual(result, [[1, 2], [3, 4]])
            mock_sampler.sample.assert_called_once_with([[1, 2]], [0, 1], False)

    def test_layerwise_sample_with_hierarchy(self):
        """layerwise_sample with hierarchy flag.
        带层级标志的 layerwise_sample。"""
        with patch.object(idx_mod, "core") as mock_core:
            mock_core.IndexWrapper.return_value = self.mock_wrapper
            mock_sampler = MagicMock()
            mock_sampler.sample.return_value = [[1, 2]]
            mock_core.IndexSampler.return_value = mock_sampler

            idx = TreeIndex("test_tree", "/path")
            idx.init_layerwise_sampler([10, 5])
            result = idx.layerwise_sample([[1, 2]], [0, 1], with_hierarchy=True)

            mock_sampler.sample.assert_called_once_with([[1, 2]], [0, 1], True)


if __name__ == "__main__":
    unittest.main()
