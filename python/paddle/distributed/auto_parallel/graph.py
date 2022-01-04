#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License


class Node:
    def __init__(self, id, **attrs):
        # Each node must has a unique id
        self._id = id
        # Attributes for Node
        self._attrs = {}
        self._attrs.update(attrs)

    @property
    def id(self):
        return self._id

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, attr_name):
        return self._attrs[attr_name]

    def __setitem__(self, attr_name, attr_value):
        self._attrs[attr_name] = attr_value

    def __contains__(self, attr_name):
        try:
            return attr_name in self._attrs
        except TypeError:
            return False

    def __str__(self):
        str = "(id: {}, attrs: {})".format(self.id, self.attrs)
        return str


class Edge:
    def __init__(self, src_id, tgt_id, **attrs):
        # The id of source node in an Edge
        self._src_id = src_id
        # The id of target node in an Edge
        self._tgt_id = tgt_id
        # Attributes for Edge
        self._attrs = {}
        self._attrs.update(attrs)

    @property
    def src_id(self):
        return self._src_id

    @property
    def tgt_id(self):
        return self._tgt_id

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, attr_name):
        return self._attrs[attr_name]

    def __setitem__(self, attr_name, attr_value):
        self._attrs[attr_name] = attr_value

    def __contains__(self, attr_name):
        try:
            return attr_name in self._attrs
        except TypeError:
            return False

    def __str__(self):
        str = ""
        str += "(src_id: {}, tgt_id: {}, attrs: {})".format(
            self.src_id, self.tgt_id, self._attrs)
        return str


class Graph:
    def __init__(self, **attrs):
        # _nodes is dict for storing the nodes of the graph.
        # The key of this dict is the node id.
        self._nodes = {}
        # _adjs is a dict of dict for storing the adjacency of the graph.
        # The key of the outer dict is the node id of the source node and
        # the key of the inner dict is the node id of the target node.
        self._adjs = {}
        # Attributes for Graph
        self._attrs = {}
        self._attrs.update(attrs)

    @property
    def nodes(self):
        return self._nodes

    @property
    def attrs(self):
        return self._attrs

    @property
    def adjs(self):
        return self._adjs

    def add_node(self, node_id, **attrs):
        if node_id is None:
            raise ValueError("None cannot be a node")
        if node_id not in self._nodes:
            node = Node(node_id, **attrs)
            self._nodes[node_id] = node
            self._adjs[node_id] = {}
        else:
            self._nodes[node_id].attrs.update(attrs)

    def add_edge(self, src_id, tgt_id, **attrs):
        # add nodes
        if src_id is None:
            raise ValueError("None cannot be a node")
        if tgt_id is None:
            raise ValueError("None cannot be a node")
        if src_id not in self._nodes:
            src_node = Node(src_id)
            self._nodes[src_id] = src_node
            self._adjs[src_id] = {}
        if tgt_id not in self._nodes:
            tgt_node = Node(tgt_id)
            self._nodes[tgt_id] = tgt_node
            self._adjs[tgt_id] = {}
        # add the edge
        edge = Edge(src_id, tgt_id, **attrs)
        self._adjs[src_id][tgt_id] = edge

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.values())

    def __getitem__(self, node_id):
        # Return the adjacency of a node
        return self._adjs[node_id]

    def __contains__(self, node_id):
        # Check whether a node in the graph
        try:
            return node_id in self._nodes
        except TypeError:
            return False

    def __str__(self):
        str = ""
        str += "**************Nodes**************\n"
        for node_id in self.nodes:
            str += "{}\n".format(self.nodes[node_id])

        str += "**************Edges**************\n"
        for src_id in self.adjs:
            str += "--------------{}--------------\n".format(src_id)
            for idx, tgt_id in enumerate(self.adjs[src_id]):
                str += "{}\n".format(self.adjs[src_id][tgt_id])

        return str
