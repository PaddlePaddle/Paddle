# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from abc import abstractmethod

from ..graph import Graph

_PATTERNS = {}
_PATTERN_MAP = {}


def register_pattern(cls):
    """Register pattern for rule-based tuner."""

    def register():
        global _PATTERNS
        global _PATTERN_MAP
        pattern = cls()
        _PATTERNS.append(pattern)
        _PATTERNS.sort(key=lambda x: -x.attrs["weights"])
        _PATTERN_MAP[pattern.name] = pattern

    register()

    return cls


class BasePattern(Graph):
    name = "base"

    def __init__(self):
        super().__init__()
        self.build()

    @abstractmethod
    def build(self):
        pass


@register_pattern
class SelfAttentionPattern(BasePattern):
    name = "self_attention"

    def __init__(self):
        super().__init__()

    def build(self):
        query = self.add_node(0, **{"type": "var"})

        # define q, k, v weight
        q_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        k_weight = self.add_node(2, **{"dim": 2, "type": "param"})
        v_weight = self.add_node(3, **{"dim": 2, "type": "param"})
        # define q, k, v matmul_v2
        q_matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})
        k_matmul_v2 = self.add_node(5, **{"type": "matmul_v2"})
        v_matmul_v2 = self.add_node(6, **{"type": "matmul_v2"})
        # define input edge
        q_x_edge = self.add_edge(
            query.id, q_matmul_v2.id, **{"input_name": "X"}
        )
        k_x_edge = self.add_edge(
            query.id, k_matmul_v2.id, **{"input_name": "X"}
        )
        v_x_edge = self.add_edge(
            query.id, v_matmul_v2.id, **{"input_name": "X"}
        )
        q_y_edge = self.add_edge(
            q_weight.id, q_matmul_v2.id, **{"input_name": "Y"}
        )
        k_y_edge = self.add_edge(
            k_weight.id, k_matmul_v2.id, **{"input_name": "Y"}
        )
        v_y_edge = self.add_edge(
            v_weight.id, v_matmul_v2.id, **{"input_name": "Y"}
        )
        # define q, k, v matmul_v2 output
        q = self.add_node(7, **{"type": "var"})
        k = self.add_node(8, **{"type": "var"})
        v = self.add_node(9, **{"type": "var"})
        # define output edge
        q_out_edge = self.add_edge(
            q_matmul_v2.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_matmul_v2.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_matmul_v2.id, v.id, **{"output_name": "Out"}
        )

        # define add weight
        q_bias = self.add_node(10, **{"dim": 1, "type": "param"})
        k_bias = self.add_node(11, **{"dim": 1, "type": "param"})
        v_bias = self.add_node(12, **{"dim": 1, "type": "param"})
        # define add
        q_add = self.add_node(13, **{"type": "elementwise_add"})
        k_add = self.add_node(14, **{"type": "elementwise_add"})
        v_add = self.add_node(15, **{"type": "elementwise_add"})
        # define add input edge
        q_x_edge = self.add_edge(q.id, q_add.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_add.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_add.id, **{"input_name": "X"})
        q_y_edge = self.add_edge(q_bias.id, q_add.id, **{"input_name": "Y"})
        k_y_edge = self.add_edge(k_bias.id, k_add.id, **{"input_name": "Y"})
        v_y_edge = self.add_edge(v_bias.id, v_add.id, **{"input_name": "Y"})
        # define add output
        q = self.add_node(16, **{"type": "var"})
        k = self.add_node(17, **{"type": "var"})
        v = self.add_node(18, **{"type": "var"})
        # define add output egde
        q_out_edge = self.add_edge(q_add.id, q.id, **{"output_name": "Out"})
        k_out_edge = self.add_edge(k_add.id, k.id, **{"output_name": "Out"})
        v_out_edge = self.add_edge(v_add.id, v.id, **{"output_name": "Out"})

        # define reshape
        q_reshape = self.add_node(19, **{"type": "reshape2"})
        k_reshape = self.add_node(20, **{"type": "reshape2"})
        v_reshape = self.add_node(21, **{"type": "reshape2"})
        # define reshape input egde
        q_x_edge = self.add_edge(q.id, q_reshape.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_reshape.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_reshape.id, **{"input_name": "X"})
        # define reshape out
        q = self.add_node(22, **{"type": "var"})
        k = self.add_node(23, **{"type": "var"})
        v = self.add_node(24, **{"type": "var"})
        # define reshape output edge
        q_out_edge = self.add_edge(q_reshape.id, q.id, **{"output_name": "Out"})
        k_out_edge = self.add_edge(k_reshape.id, k.id, **{"output_name": "Out"})
        v_out_edge = self.add_edge(v_reshape.id, v.id, **{"output_name": "Out"})

        # define transpose
        q_transpose = self.add_node(25, **{"type": "transpose2"})
        k_transpose = self.add_node(26, **{"type": "transpose2"})
        v_transpose = self.add_node(27, **{"type": "transpose2"})
        # define transpose input edge
        q_x_edge = self.add_edge(q.id, q_transpose.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_transpose.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_transpose.id, **{"input_name": "X"})
        # define transpose output
        q = self.add_node(28, **{"type": "var"})
        k = self.add_node(29, **{"type": "var"})
        v = self.add_node(30, **{"type": "var"})
        # define transpose output edege
        q_out_edge = self.add_edge(
            q_transpose.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_transpose.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_transpose.id, v.id, **{"output_name": "Out"}
        )

        # define matmul
        matmul = self.add_node(31, **{"type": "matmul"})
        # define matmul input edge
        x_edge = self.add_edge(q.id, matmul.id, **{"input_name": "X"})
        y_edge = self.add_edge(k.id, matmul.id, **{"input_name": "Y"})
        # define matmul output
        out = self.add_node(32, **{"type": "var"})
        # define matmul output edge
        out_edge = self.add_edge(matmul.id, out.id, **{"output_name": "Out"})

        # define add y
        attention_mask = self.add_node(33, **{"type": "data"})
        # define add
        add = self.add_node(34, **{"type": "elementwise_add"})
        # define add input edge
        x_edge = self.add_edge(out.id, add.id, **{"input_name": "X"})
        y_edge = self.add_edge(attention_mask.id, add.id, **{"input_name": "Y"})
        # define add output
        out = self.add_node(35, **{"type": "var"})
        # define add output egde
        out_edge = self.add_edge(add.id, out.id, **{"output_name": "Out"})

        # define softmax
        softmax = self.add_node(36, **{"type": "softmax"})
        # define input edge
        input_edge = self.add_edge(out.id, softmax.id, **{"input_name": "X"})
        # define softmax output
        out = self.add_node(37, **{"type": "var"})
        # define softmax output edge
        output_edge = self.add_edge(
            softmax.id, out.id, **{"output_name": "Out"}
        )

        # define matmul_v2
        matmul_v2 = self.add_node(38, **{"type": "matmul_v2"})
        # define input edge
        x_edge = self.add_edge(out.id, matmul_v2.id, **{"input_name": "X"})
        y_edge = self.add_edge(v.id, matmul_v2.id, **{"input_name": "Y"})
        # define output
        out = self.add_node(39, **{"type": "var"})
        # define output edge
        out_edge = self.add_edge(matmul_v2.id, out.id, **{"output_name": "Out"})

        # define transpose
        transpose = self.add_node(40, **{"type": "transpose2"})
        # define transpose input edge
        x_edge = self.add_edge(out.id, transpose.id, **{"input_name": "X"})
        # define transpose output
        out = self.add_node(41, **{"type": "var"})
        # define transpose output edege
        out_edge = self.add_edge(transpose.id, out.id, **{"output_name": "Out"})

        # define reshape
        reshape = self.add_node(42, **{"type": "reshape2"})
        # define reshape input egde
        x_edge = self.add_edge(out.id, reshape.id, **{"input_name": "X"})
        # define reshape out
        out = self.add_node(43, **{"type": "var"})
        # define reshape output edge
        out_edge = self.add_edge(reshape.id, out.id, **{"output_name": "Out"})

        # define matmul_v2 weight
        y = self.add_node(44, **{"type": "param", "dim": 2})
        # define matmul_v2
        matmul_v2 = self.add_node(45, **{"type": "matmul_v2"})
        # define input edge
        x_edge = self.add_edge(out.id, matmul_v2.id, **{"input_name": "X"})
        y_edge = self.add_edge(y.id, matmul_v2.id, **{"input_name": "Y"})
        # define output
        out = self.add_node(46, **{"type": "var"})
        # define output edge
        out_edge = self.add_edge(matmul_v2.id, out.id, **{"output_name": "Out"})

        # define add weight
        bias = self.add_node(47, **{"dim": 1, "type": "param"})
        # define add
        add = self.add_node(48, **{"type": "elementwise_add"})
        # define add input edge
        x_edge = self.add_edge(out.id, add.id, **{"input_name": "X"})
        y_edge = self.add_edge(bias.id, add.id, **{"input_name": "Y"})
        # define add output
        out = self.add_node(49, **{"type": "var"})
        # define add output egde
        out_edge = self.add_edge(add.id, out.id, **{"output_name": "Out"})

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp0mp1": {
                0: [0, -1, -1],
                1: [-1, 1],
                2: [-1, 1],
                3: [-1, 1],
                44: [1, -1],
            },
            "dp1mp0": {
                0: [1, -1, -1],
                1: [-1, 0],
                2: [-1, 0],
                3: [-1, 0],
                44: [0, -1],
            },
            "mp": {1: [-1, 0], 2: [-1, 0], 3: [-1, 0], 44: [0, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 4


class GraphUtil:
    @staticmethod
    def convert_to_graph(block):
        """Convert ops to graph."""
        graph = Graph()
        graph.attrs["var_to_id"] = {}  # {var_name: node_id}
        graph.attrs["id_to_var_desc_id"] = {}  # {node_id: var_desc_id}
        graph.attrs["id_to_var_name"] = {}
        graph.attrs["op_to_id"] = {}  # {op_id: node_id}
        graph.attrs["id_to_op"] = {}  # {node_id: op}

        ops = block.ops
        node_id = -1
        for op in ops:
            attrs = op.all_attrs()
            attrs["type"] = op.type
            node_id += 1

            # create op node
            op_node = graph.add_node(node_id, **attrs)
            graph.attrs["op_to_id"][op.desc.id()] = op_node.id
            graph.attrs["id_to_op"][op_node.id] = op
            graph._attr_to_nodes[op_node.id] = {}
            for input_name in op.input_names:
                graph._attr_to_nodes[op_node.id][input_name] = []
                for var_name in op.input(input_name):
                    if var_name not in graph.attrs["var_to_id"]:
                        # create var node
                        node_id += 1
                        var_node = graph.add_node(node_id)
                        var = block._var_recursive(var_name)
                        if var.is_parameter:
                            var_node.attrs["type"] = "param"
                            var_node.attrs["dim"] = len(var.shape)
                        elif var.is_data:
                            var_node.attrs["type"] = "data"
                            var_node.attrs["dim"] = len(var.shape)
                        else:
                            var_node.attrs["type"] = "var"
                        graph.attrs["var_to_id"][var_name] = var_node.id
                        graph.attrs["id_to_var_desc_id"][
                            var_node.id
                        ] = var.desc.original_id()
                        graph.attrs["id_to_var_name"][var_node.id] = var_name
                    else:
                        var_node_id = graph.attrs["var_to_id"][var_name]
                        var_node = graph._nodes[var_node_id]

                    # create edge that input -> op
                    input_edge = graph.add_edge(var_node.id, op_node.id)
                    input_edge.attrs["input_name"] = input_name
                    graph._attr_to_nodes[op_node.id][input_name].append(
                        var_node
                    )

                for output_name in op.output_names:
                    graph._attr_to_nodes[op_node.id][output_name] = []
                    for var_name in op.output(output_name):
                        if var_name not in graph.attrs["var_to_id"]:
                            # create var node
                            node_id += 1
                            var_node = graph.add_node(node_id)
                            var = block._var_recursive(var_name)
                            if var.is_parameter:
                                var_node.attrs["type"] = "param"
                            else:
                                var_node.attrs["type"] = "var"
                            graph.attrs["var_to_id"][var_name] = var_node.id
                            graph.attrs["id_to_var_desc_id"][
                                var_node.id
                            ] = var.desc.original_id()
                            graph.attrs["id_to_var_name"][
                                var_node.id
                            ] = var_name
                        else:
                            var_node_id = graph.attrs["var_to_id"][var_name]
                            var_node = graph._nodes[var_node_id]

                        # create edge that op -> output
                        output_edge = graph.add_edge(op_node.id, var_node.id)
                        output_edge.attrs["output_name"] = output_name

                        graph._attr_to_nodes[op_node.id][output_name].append(
                            var_node
                        )

        return graph

    @staticmethod
    def match_pattern(pattern, graph):
        def _is_op_node(node):
            """Judge whether node is op node"""
            if node.attrs["type"] not in ["var", "param", "data"]:
                return True

            return False

        def _compare_op_node(src, tgt):
            """Compare whether two op nodes are equal"""
            if src.attrs["type"] != tgt.attrs["type"]:
                return False

            return True

        def _compare_var_node(src, tgt):
            """Compare whether two var nodes are equal"""
            for key in src.attrs:
                if key not in tgt.attrs:
                    return False
                if src.attrs[key] != tgt.attrs[key]:
                    return False

            return True

        def _match_core(src_node, tgt_node):
            nonlocal not_matched
            # do not support one input name or output name corresponding to multiple vars
            if not_matched:
                return

            if _is_op_node(src_node):
                # compare op node whether equal
                if not _compare_op_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # input var nodes
                src_input_nodes = src_reverse_adjs[src_node.id]
                for node in src_input_nodes:
                    # has visited
                    if node.id in result:
                        continue
                    edge = src_edges[node.id][src_node.id]
                    input_name = edge.attrs["input_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        input_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

                # output var nodes
                src_output_node_ids = src_edges[src_node.id].keys()
                for node_id in src_output_node_ids:
                    # has visited
                    if node_id in result:
                        continue
                    node = src_nodes[node_id]
                    edge = src_edges[src_node.id][node_id]
                    output_name = edge.attrs["output_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        output_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

            else:
                # compare var nodes whether equal
                if not _compare_var_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # as input for op node
                src_as_input_node_ids = src_edges[src_node.id].keys()
                for node_id in src_as_input_node_ids:
                    if node_id in result:
                        continue

                    src_edge = src_edges[src_node.id][node_id]
                    input_name = src_edge.attrs["input_name"]
                    compare_node_ids = tgt_edges[tgt_node.id].keys()

                    compare_node = None
                    for compare_node_id in compare_node_ids:
                        edge = tgt_edges[tgt_node.id][compare_node_id]
                        if (
                            edge.attrs["input_name"] == input_name
                            and compare_node_id not in result.values()
                        ):
                            compare_node = tgt_nodes[compare_node_id]
                            break

                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node_id], compare_node)

                # as output for op node
                src_as_output_nodes = src_reverse_adjs[src_node.id]
                for node in src_as_output_nodes:
                    if node.id in result:
                        continue

                    src_edge = src_edges[node.id][src_node.id]
                    output_name = src_edge.attrs["output_name"]

                    compare_nodes = tgt_reverse_adjs[tgt_node.id]

                    compare_node = None
                    for item in compare_nodes:
                        node_id = item.id
                        edge = tgt_edges[node_id][tgt_node.id]
                        if edge.attrs["output_name"] == output_name:
                            compare_node = tgt_nodes[node_id]
                            break
                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node.id], compare_node)

        results = []
        matched_ids = set()
        matched_op_node_ids = set()
        result = {}
        src_nodes = pattern.nodes
        src_edges = pattern._adjs
        src_reverse_adjs = pattern._reverse_adjs

        tgt_nodes = graph.nodes
        tgt_edges = graph._adjs
        tgt_reverse_adjs = graph._reverse_adjs
        tgt_attr_to_nodes = graph._attr_to_nodes

        # starts with a op node
        src_start_node = None
        for node_id in src_nodes:
            node = src_nodes[node_id]
            if node.attrs["type"] not in ["var", "param", "data"]:
                src_start_node = node
                break
        assert src_start_node is not None

        for node_id in tgt_nodes:
            node = tgt_nodes[node_id]
            if node.attrs["type"] == src_start_node.attrs["type"]:
                not_matched = False
                _match_core(src_start_node, node)
                if not not_matched:
                    need_to_append = True
                    for value in result.values():
                        if value in matched_op_node_ids:
                            result = {}
                            need_to_append = False
                            break
                    if need_to_append:
                        results.append(result)
                        for value in result.values():
                            matched_ids.add(value)
                            if value in graph.attrs["id_to_op"].keys():
                                matched_op_node_ids.add(value)
                        result = {}
                else:
                    not_matched = False
                    result = {}

        return results, matched_ids

    @staticmethod
    def match_all_patterns(graph):
        matched_results = (
            {}
        )  # {"pattern_name": [{0: graph_node}, {0: graph_node}]}
        matched_ids = set()
        for pattern in _PATTERNS:
            has_matched = False
            results, matched = GraphUtil.match_pattern(pattern, graph)
            for id in matched:
                if id in matched_ids:
                    has_matched = True
                    break
            if not has_matched:
                for item in matched:
                    matched_ids.add(item)
                matched_results[pattern.name] = results

        return matched_results


class OperatorClusteringUtil:
    common_starts = ["layer_norm", "matmul_v2", "matmul"]

    @staticmethod
    def get_ranks(seq):
        """Get rank array of the given seq by doubled algorithm."""
        ordered_seq = sorted(list(set(seq)))
        item_to_rank = {item: idx for idx, item in enumerate(ordered_seq)}
        inter_ranks = [item_to_rank[item] for item in seq]

        length = len(inter_ranks)
        power = 0
        interval = 2**power
        while interval < length:
            for idx, item in enumerate(inter_ranks):
                if idx + interval >= length:
                    inter_ranks[idx] = [item, -1]
                else:
                    inter_ranks[idx] = [item, inter_ranks[idx + interval]]

            tmp = []
            for item in inter_ranks:
                if item not in tmp:
                    tmp.append(item)
            tmp.sort(key=lambda x: (x[0], x[1]))
            item_to_rank = {}
            for idx, val in enumerate(tmp):
                key = ",".join(str(item) for item in val)
                item_to_rank[key] = idx

            inter_ranks = [
                item_to_rank[",".join(str(val) for val in item)]
                for item in inter_ranks
            ]
            power += 1
            interval = 2**power

        return inter_ranks

    @staticmethod
    def get_suffixes(ranks):
        """Get suffix array by the given rank array."""
        suffixes = [0 for idx in range(len(ranks))]
        for idx, item in enumerate(ranks):
            suffixes[item] = idx
        return suffixes

    @staticmethod
    def get_heights(suffixes, seq):
        """Get height array by the suffix array and seq"""
        heights = [-1 for i in range(len(suffixes))]
        for i in range(1, len(seq)):
            x = seq[suffixes[i - 1] :]
            y = seq[suffixes[i] :]
            max_len = len(x) if len(x) > len(y) else len(y)
            same_count = 0
            for j in range(max_len):
                if j >= len(x) or j >= len(y):
                    break
                else:
                    if x[j] == y[j]:
                        same_count += 1
                    else:
                        break
            heights[i] = same_count

        return heights

    @staticmethod
    def get_longest_repeated_sub_seq(suffixes, heights, seq):
        """Get longest repeated sub sequence by suffix array algorithm."""
        length = len(seq)
        if length <= 1:
            return None
        k = length // 2
        height_groups = []
        longest_sub_seq = None
        longest_sub_seqs = []

        while k >= 2:
            height_group = []
            for i in range(1, len(heights)):
                if heights[i] >= k:
                    if i == 1:
                        height_group.append(0)
                    height_group.append(i)
                else:
                    if i == 1:
                        height_groups.append([0])
                        height_group = [i]
                    else:
                        height_groups.append(height_group)
                        height_group = [i]

            if height_group:
                height_groups.append(height_group)

            for height_group in height_groups:
                suffix_group = []
                index_group = []
                for idx in height_group:
                    suffix_group.append(idx)
                    index_group.append(suffixes[idx])

                max_index = max(index_group)
                min_index = min(index_group)
                if max_index - min_index >= k:
                    longest_sub_seq = seq[min_index : min_index + k]
                    if (
                        longest_sub_seq[0]
                        in OperatorClusteringUtil.common_starts
                    ):
                        return longest_sub_seq
            if longest_sub_seq is not None:
                return longest_sub_seq

            k -= 1
            height_groups = []

        return longest_sub_seq

    @staticmethod
    def get_decomposed_sub_seq(seq):
        """Get decomposed sub seq s by seq S such as s * R = S."""
        if not seq:
            return seq

        decomposed_sub_seq = seq
        seq_len = len(seq)
        if seq_len == 1:
            return decomposed_sub_seq
        else:
            for interval in range(2, seq_len + 1):
                if seq_len % interval == 0:
                    repeated_times = seq_len // interval
                    decomposed_sub_seq = seq[0:interval]
                    decomposed = True
                    for j in range(1, repeated_times + 1):
                        sub_seq = seq[interval * (j - 1) : interval * j]
                        if sub_seq != decomposed_sub_seq:
                            decomposed = False
                            break
                    if decomposed:
                        return decomposed_sub_seq

        return decomposed_sub_seq

    @staticmethod
    def replace_by_decomposed_seq(sub_seq, seq):
        """Replace seq by sub seq."""
        if not sub_seq:
            return seq

        result = []
        sub_seq_len = len(sub_seq)
        i = 0
        while i < len(seq):
            if seq[i : i + sub_seq_len] == sub_seq:
                result.append(seq[i : i + sub_seq_len])
                i += sub_seq_len
            else:
                result.append(seq[i])
                i += 1

        return result

    @staticmethod
    def stop_replace(seq):
        for item in seq:
            if not isinstance(item, list):
                return False
        return True


class RuleBasedTuner:
    def __init__(self, dist_context, mode="train"):
        self._dist_context = dist_context
        self._mode = mode

    def cluster_operators(self):
        """
        Group operators to layers.

        Args:
            ops (list): A operator list.

        Returns:
            List: The list contains the list of operators which belong to the same layer.
        """
        ops = self._dist_context._serial_main_program.global_block().ops
        seq = [op.type for op in ops]

        while not OperatorClusteringUtil.stop_replace(seq):
            to_replace_seq = []
            to_replace_idxes = []
            has_append = False
            for idx, item in enumerate(seq):
                if not isinstance(item, list):
                    has_append = True
                    to_replace_seq.append(item)
                    to_replace_idxes.append(idx)
                elif isinstance(seq, list) and not has_append:
                    continue
                elif isinstance(seq, list) and has_append:
                    break

            ranks = OperatorClusteringUtil.get_ranks(to_replace_seq)
            suffixes = OperatorClusteringUtil.get_suffixes(ranks)
            heights = OperatorClusteringUtil.get_heights(
                suffixes, to_replace_seq
            )
            longest_sub_seq = (
                OperatorClusteringUtil.get_longest_repeated_sub_seq(
                    suffixes, heights, to_replace_seq
                )
            )
            has_merged = False
            if longest_sub_seq is None:
                for i in range(to_replace_idxes[-1] + 1, len(seq)):
                    if isinstance(seq[i], list):
                        seq[i] = to_replace_seq + seq[i]
                        has_merged = True
                        break
                if not has_merged:
                    for i in range(to_replace_idxes[0] - 1, -1, -1):
                        if isinstance(seq[i], list):
                            seq[i].extend(to_replace_seq)
                            has_merged = True
                            break
                if not has_merged:
                    seq = [to_replace_seq]
                    break

            decomposed_sub_seq = OperatorClusteringUtil.get_decomposed_sub_seq(
                longest_sub_seq
            )
            to_replace_seq = OperatorClusteringUtil.replace_by_decomposed_seq(
                decomposed_sub_seq, to_replace_seq
            )
            result = seq[: to_replace_idxes[0]]
            if not has_merged:
                result.extend(to_replace_seq)
            result.extend(seq[to_replace_idxes[-1] + 1 :])
            seq = result

        layers = []
        idx = 0
        for groups in seq:
            layer = []
            for op in groups:
                layer.append(ops[idx])
                idx += 1
            layers.append(layer)

        return layers
