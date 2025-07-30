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

from __future__ import annotations

from typing import Any

from paddle.base import core

__all__ = []


class Index:
    def __init__(self, name: str) -> None:
        self._name = name


class TreeIndex(Index):
    def __init__(self, name: str, path: str) -> None:
        super().__init__(name)
        self._wrapper = core.IndexWrapper()
        self._wrapper.insert_tree_index(name, path)
        self._tree = self._wrapper.get_tree_index(name)
        self._height = self._tree.height()
        self._branch = self._tree.branch()
        self._total_node_nums = self._tree.total_node_nums()
        self._emb_size = self._tree.emb_size()
        self._layerwise_sampler = None

    def height(self) -> int:
        return self._height

    def branch(self) -> int:
        return self._branch

    def total_node_nums(self) -> int:
        return self._total_node_nums

    def emb_size(self) -> int:
        return self._emb_size

    def get_all_leaves(self) -> list[Any]:
        return self._tree.get_all_leaves()

    def get_nodes(self, codes: list[int]) -> list[Any]:
        return self._tree.get_nodes(codes)

    def get_layer_codes(self, level: int) -> list[int]:
        return self._tree.get_layer_codes(level)

    def get_travel_codes(self, id: int, start_level: int = 0) -> list[int]:
        return self._tree.get_travel_codes(id, start_level)

    def get_ancestor_codes(self, ids: list[int], level: int) -> list[int]:
        return self._tree.get_ancestor_codes(ids, level)

    def get_children_codes(self, ancestor: int, level: int) -> list[int]:
        return self._tree.get_children_codes(ancestor, level)

    def get_travel_path(self, child: int, ancestor: int) -> list[int]:
        res = []
        while child > ancestor:
            res.append(child)
            child = int((child - 1) / self._branch)
        return res

    def get_pi_relation(self, ids: list[int], level: int) -> dict[int, int]:
        codes = self.get_ancestor_codes(ids, level)
        return dict(zip(ids, codes))

    def init_layerwise_sampler(
        self,
        layer_sample_counts: list[int],
        start_sample_layer: int = 1,
        seed: int = 0,
    ) -> None:
        assert self._layerwise_sampler is None
        self._layerwise_sampler = core.IndexSampler("by_layerwise", self._name)
        self._layerwise_sampler.init_layerwise_conf(
            layer_sample_counts, start_sample_layer, seed
        )

    def layerwise_sample(
        self,
        user_input: list[list[int]],
        index_input: list[int],
        with_hierarchy: bool = False,
    ) -> list[list[int]]:
        if self._layerwise_sampler is None:
            raise ValueError("please init layerwise_sampler first.")
        return self._layerwise_sampler.sample(
            user_input, index_input, with_hierarchy
        )
