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

from paddle.fluid.core import IndexWrapper, TreeIndex


class Index(object):
    def __init__(self, name):
        self._name = name


class GraphIndex(Index):
    def __init__(self, name):
        super(GraphIndex, self).__init__(name)


class TreeIndex(Index):
    def __init__(self, name, path):
        super(TreeIndex, self).__init__(name)
        self._wrapper = IndexWrapper()
        self._wrapper.insert_tree_index(name, path)
        self._tree = self._wrapper.get_tree_index(name)
        self._height = self._tree.height()
        self._branch = self._tree.branch()

    def get_nodes_given_level(self, level, ret_code=False):
        return self._tree.get_nodes_given_level(level, ret_code)

    def get_parent_path(self, ids, start_level=0, ret_code=False):
        return self._tree.get_parent_path(ids, start_level, ret_code)

    def height(self):
        return self._height

    def branch(self):
        return self._branch

    def total_node_nums(self):
        return self._tree.total_node_nums()

    def get_ancestor_given_level(self, ids, level, ret_code=False):
        return self._tree.get_ancestor_given_level(ids, level, ret_code)

    def get_all_items(self):
        return self._tree.get_all_items()

    def get_pi_relation(self, ids, level):
        return self._tree.get_pi_relation(ids, level)

    def get_children_given_ancestor_and_level(self,
                                              ancestor,
                                              level,
                                              ret_code=True):
        return self._tree.get_children_given_ancestor_and_level(ancestor, level,
                                                                ret_code)

    def get_travel_path(self, child, ancestor, ret_code=False):
        if ret_code == False:
            return self._tree.get_travel_path(child, ancestor)
        else:
            res = []
            while (child > ancestor):
                res.append(child)
                child = int((child - 1) / self._branch)

            return res

    def tree_max_node(self):
        return self._tree.tree_max_node()
