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
# limitations under the License.

import copy
import queue
from enum import Enum

import numpy as np

import paddle
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.framework import core

SUCC = 0  # successor
PRED = 1  # predecessor


class CostNodeType(Enum):
    DEFAULT = 0
    COMPUTATION = 1
    COMMUNICATION = 2
    VARIABLE = 3
    MERGED = 4
    NOP = 5


class Cost:
    def __init__(self):
        self.runtime = None
        self.static_mem = None
        self.peak_mem = None


class CostModelMode(Enum):
    DEFAULT = 0
    BENCHMARKING = 1  # costs based on trial runs
    ANALYSIS = 2  # costs based on analysis
    MIXED = 3


class CostNode:
    def __init__(self, node, node_type, id=None):
        self.id = id
        self.node = node
        self.type = node_type
        self._cost = 0
        self.is_optim = False
        self.is_bwd = False

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        if cost < 0:
            raise ValueError('Cost must be above 0.')
        self._cost = cost


class MergedOpsCostNode(CostNode):
    def __init__(self, node_type, id=None, base_node_list=None, is_bwd=False):
        super().__init__(None, node_type, id)
        self.node_list = base_node_list
        self.is_bwd = is_bwd


class CommOpCostNode(CostNode):
    def __init__(
        self, node, node_type, id=None, comm_node_list=None, is_bwd=False
    ):
        super().__init__(node, node_type, id)
        self.node_list = comm_node_list
        self.ranks = []
        self.comm_type = node.type
        self.is_bwd = is_bwd

    def set_ranks(self, ranks):
        self.ranks = ranks

    def set_shapes(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def init_comm_cost(self, cluster=None):
        # ref: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
        # should get from `cluster`
        BANDWIDTH = 32 * 1024 / 1000  # MB/ms, V100 PCIe
        num_ranks = len(self.ranks)
        comm_volume = np.prod(self.input_shape) * 4

        if 'allreduce' in self.comm_type:
            self._cost = comm_volume / (
                BANDWIDTH * num_ranks / (2 * (num_ranks - 1))
            )
        elif 'gather' in self.comm_type:
            self._cost = comm_volume / (BANDWIDTH * num_ranks / (num_ranks - 1))
        elif 'broadcast' in self.comm_type:
            self._cost = comm_volume / BANDWIDTH
        elif 'send' in self.comm_type or 'recv' in self.comm_type:
            self._cost = comm_volume / BANDWIDTH
        else:
            self._cost = 0


class TensorCostNode(CostNode):
    def __init__(
        self,
        node,
        node_type,
        id=None,
        base_node_list=None,
        batch_size=None,
        shared_node_id=None,
    ):
        super().__init__(node, node_type, id)
        if node.name == "create_py_reader_0" or node.name == "double_buffer_0":
            self.shape = [2, 2]
            self.dtype = paddle.float32
        else:
            self.shape = node.shape
            self.dtype = node.dtype
        self.dtype_factor = 1
        self.persistable = None
        self.shared_node_id = shared_node_id
        if self.dtype == paddle.float32 or node.dtype == paddle.int32:
            self.dtype_factor *= 4
        elif node.dtype == paddle.int64:
            self.dtype_factor *= 8
        elif node.dtype == paddle.uint8:
            self.dtype_factor = 1
        else:
            self.dtype_factor = 2
            # raise NotImplementedError("{} not counted".format(node.dtype))
        self.batch_size = None
        if batch_size is not None:
            self.batch_size = batch_size

    def get_size(self):
        p = 1
        for i in self.node.shape:
            if i == -1:  # deal with placeholder
                assert self.batch_size is not None, "Batch size not decided."
                i = self.batch_size
            p *= i
        return p


class CompOpCostNode(CostNode):
    def __init__(self, node, node_type, id=None, is_bwd=False, is_optim=False):
        super().__init__(node, node_type, id)
        self.is_bwd = is_bwd
        self.is_optim = is_optim

    def init_comp_cost(self, cost_data):
        # TODO: improve base.CostModel for more specific cost_data
        op_id = self.node.desc.id()
        if op_id in cost_data.keys():
            self.cost = cost_data[op_id]
        else:
            self.cost = 0.0


class PipeEvent:
    def __init__(self, stage_id, event_name, duration, start_time=-1):
        self.stage_id = stage_id
        self.name = event_name
        self.duration = duration
        self.s_time = start_time
        self.e_time = -1


class CostModel:
    def __init__(
        self,
        mode=CostModelMode.BENCHMARKING,
        cluster=None,
        batch_size=1,
        microbatch_num=1,
        opcall_overhead=0,
        standalone_cost_data=None,
        pipeline_config=None,
    ):
        self.mode = mode

        # parameters
        self.opcall_overhead = opcall_overhead
        self.batch_size = batch_size
        self.microbatch_num = microbatch_num

        self.nodes = {}  # name -> node

        self.origin_graph = {}  # original graph
        self.op_graph = {}  # op graph (no variables nodes)
        self.runtime_graph = {}  # runtime graph, for simulation

        self.cluster = cluster
        self.cost_data = standalone_cost_data
        self.pp2rank = pipeline_config
        if self.pp2rank is not None:
            self.rank2pp = {}
            for stage_idx, ranks in enumerate(self.pp2rank):
                for rank in ranks:
                    self.rank2pp[rank] = stage_idx
        else:
            self.rank2pp = None

        self.ring2rank = {}

        self.fwd_time = []
        self.bwd_time = []
        self.optim_time = []

    def _parse_sub_program(self, program, nodes, graph, cost_data, sub_idx):
        assert (
            len(program.blocks) == 1
        ), "Program more than 1 block not supported."
        block = program.blocks[0]

        var_id = "lod_tensor_blocking_queue_0"
        new_var = program.global_block().create_var(
            name=var_id,
            dtype=paddle.float32,
            type=core.VarDesc.VarType.DENSE_TENSOR,
        )
        nodes[var_id] = TensorCostNode(
            new_var, CostNodeType.VARIABLE, "lod_tensor_blocking_queue_0"
        )
        for var in block.vars.values():
            var_id = var.name
            # if var.name == "create_py_reader_0" or var.name == "double_buffer_0":
            #     continue
            nodes[var_id] = TensorCostNode(var, CostNodeType.VARIABLE, var_id)
            graph[var_id] = [[], []]

        for op in block.ops:
            op_id = op.type + "_" + str(op.idx)
            if (
                op.type.startswith('c_')
                or op.type.startswith('send')
                or op.type.startswith('recv')
            ):
                is_bwd = False
                if (
                    op.type.startswith('c_')
                    and op.type != "c_sync_calc_stream"
                    and not op.type.startswith('c_embedding')
                ):
                    ring_id = op.attr('ring_id')
                    if ring_id not in self.ring2rank:
                        self.ring2rank[ring_id] = set()
                    self.ring2rank[ring_id].add(sub_idx)
                    is_bwd = '@GRAD' in op.output('Out')[0]
                elif op.type.startswith('recv'):
                    is_bwd = '@GRAD' in op.output('Out')[0]
                elif op.type.startswith('send'):
                    is_bwd = '@GRAD' in op.input('X')[0]
                op_node = CommOpCostNode(
                    op, CostNodeType.COMMUNICATION, op_id, is_bwd
                )
            else:
                is_bwd = (
                    int(op.attr('op_role')) == int(OpRole.Backward)
                ) or "@GRAD" in op.input_arg_names
                is_optim = 'LearningRate' in op.input_names
                op_node = CompOpCostNode(
                    op, CostNodeType.COMPUTATION, op_id, is_bwd, is_optim
                )
                op_node.init_comp_cost(cost_data)

            nodes[op_id] = op_node
            graph[op_id] = [[], []]

            comm_input_shape = [0]
            comm_output_shape = [0]
            for i in range(len(op.input_names)):
                try:
                    var_id = op.input(op.input_names[i])[0]
                    var_node = nodes[var_id]
                    graph[op_id][PRED].append(var_node.id)
                    graph[var_id][SUCC].append(op_node.id)
                    comm_input_shape = var_node.shape
                except:
                    continue

            for i in range(len(op.output_names)):
                try:
                    var_id = op.output(op.output_names[i])[0]
                    var_node = nodes[var_id]
                    graph[op_id][SUCC].append(var_node.id)
                    graph[var_id][PRED].append(op_node.id)
                    comm_output_shape = var_node.shape
                except:
                    continue
            if op_node.type == CostNodeType.COMMUNICATION:
                op_node.set_shapes(comm_input_shape, comm_output_shape)

        # resolve hazard: rename the r/w hazard variable nodes to ensure self.origin_graph is a DAG
        new_var_dict = {}
        for node_id, node in nodes.items():
            if node.type == CostNodeType.VARIABLE and node.node.persistable:
                write_op_cnt = 0
                for pred_id in graph[node_id][PRED]:
                    pred = nodes[pred_id]
                    if pred.type == CostNodeType.COMPUTATION and (
                        pred_id in graph[node_id][SUCC]
                    ):
                        graph[pred_id][SUCC].remove(node_id)
                        graph[node_id][PRED].remove(pred_id)

                        write_op_cnt += 1
                        new_var_id = node_id + f'_write_{write_op_cnt}'
                        new_var = TensorCostNode(
                            node.node,
                            CostNodeType.VARIABLE,
                            new_var_id,
                            shared_node_id=node_id,
                        )

                        graph[new_var_id] = [[], []]
                        graph[pred_id][SUCC].append(new_var_id)
                        graph[new_var_id][PRED].append(pred_id)

                        new_var_dict[new_var_id] = new_var
        for k, v in new_var_dict.items():
            nodes[k] = v
        return nodes

    def parse_program(self, distributed_program):
        self.distributed_program = distributed_program
        self.total_rank = len(self.distributed_program)
        sub_prog_cnt = len(distributed_program)
        self.nodes = [] * sub_prog_cnt
        self.origin_graph = [] * sub_prog_cnt  # original graph
        self.op_graph = [] * sub_prog_cnt  # op graph (no variables nodes)
        self.runtime_graph = [] * sub_prog_cnt  # runtime graph, for simulation

        for sub_idx, sub_prog in enumerate(distributed_program):
            self.nodes.append({})
            self.origin_graph.append({})
            self.op_graph.append({})
            self.runtime_graph.append({})
            self._parse_sub_program(
                sub_prog,
                self.nodes[sub_idx],
                self.origin_graph[sub_idx],
                self.cost_data[
                    0 if self.rank2pp is None else self.rank2pp[sub_idx]
                ],
                sub_idx,
            )
        return self.nodes

    def _find_succ_op(self, node_id, sub_idx=0):
        succ_ops_id = []
        for succ_id in self.origin_graph[sub_idx][node_id][SUCC]:
            succ = self.nodes[sub_idx][succ_id]
            if (
                succ.type == CostNodeType.COMMUNICATION
                or succ.type == CostNodeType.COMPUTATION
            ):
                succ_ops_id.append(succ_id)
            elif succ.type == CostNodeType.VARIABLE:
                succ_ops_id = succ_ops_id + self._find_succ_op(succ_id, sub_idx)
            else:
                raise NotImplementedError(
                    f'This type of node not supported yet:{succ.type}'
                )
        return succ_ops_id

    def build_op_graph(self):
        for sub_idx in range(self.total_rank):
            op_nodes_id = []
            for node_id, node in self.nodes[sub_idx].items():
                if node.type == CostNodeType.VARIABLE:
                    continue
                self.op_graph[sub_idx][node_id] = [[], []]
                op_nodes_id.append(node_id)
            for op_id in op_nodes_id:
                succ_nodes_id = self._find_succ_op(op_id, sub_idx)

                self.op_graph[sub_idx][op_id][SUCC] = succ_nodes_id
                for succ_id in succ_nodes_id:
                    self.op_graph[sub_idx][succ_id][PRED].append(op_id)

    def build_runtime_graph(self):
        self.runtime_graph = copy.deepcopy(self.op_graph)

    def eliminate_multi_edges(self, graph=None):
        for node_id, edges in graph.items():
            graph[node_id][PRED] = list(set(edges[PRED]))
            graph[node_id][SUCC] = list(set(edges[SUCC]))

    def merge_comm(self):
        for sub_idx in range(self.total_rank):
            for node_id, edges in self.op_graph[sub_idx].items():
                node = self.nodes[sub_idx][node_id]
                if (
                    node_id.startswith('c_')
                    and not node.id.startswith("c_sync_calc_stream")
                    and not node.id.startswith('c_embedding')
                ):
                    ring_id = node.node.attr('ring_id')
                    node.set_ranks(list(self.ring2rank[ring_id]))
                    node.init_comm_cost(self.cluster)
                elif node_id.startswith('send') or node_id.startswith('recv'):
                    peer_rank = node.node.attr('peer')
                    node.set_ranks([sub_idx, peer_rank])
                    node.init_comm_cost(self.cluster)
                else:
                    pass  # Not communication op

    def _merge_node(self, to_merge_node_list, merge_type='linear', nodes=None):
        nodes_list = []
        node_cost = 0
        for node in to_merge_node_list:
            if isinstance(node, MergedOpsCostNode):
                nodes_list += node.node_list
            else:
                nodes_list.append(node.id)
            if merge_type == 'linear':
                node_cost += node.cost
            elif merge_type == 'branch':
                node_cost = max(node_cost, node.cost)
            else:
                raise NotImplementedError(
                    f'This type of merging is not supported:{merge_type}'
                )
        merged_node_id = 'merged_' + str(len(nodes))
        is_bwd = to_merge_node_list[0].is_bwd
        merged_node = MergedOpsCostNode(
            CostNodeType.MERGED,
            id=merged_node_id,
            base_node_list=nodes_list,
            is_bwd=is_bwd,
        )
        merged_node.cost = node_cost
        return merged_node_id, merged_node

    def merge_linear(self):
        r'''
        This method does the following:
        If X depends on Y only, they must be run sequentially.
            [ e.g. A ->- C ->- D   D and E depends on C only.]
            [      B ->-/ \->- E   C depends on A and B.     ]
        We merge X and Y into a new node and sum up their cost time.
        '''
        cnt = 0
        for sub_idx in range(self.total_rank):
            cnt += self._merge_linear(
                self.nodes[sub_idx], self.runtime_graph[sub_idx], is_bwd=False
            )
            cnt += self._merge_linear(
                self.nodes[sub_idx], self.runtime_graph[sub_idx], is_bwd=True
            )
        return cnt

    def merge_branch(self):
        r'''
        This method does the following:
        If a node has more than one successor, there is *branch*.
            [ e.g. A ->- B ->- D                                       ]
            [       \->- C ->- / , B and C can be run at the same time ]
            case 1: if B or C is null (or D is directly dependent on A),
                    it's equivalent to A->C->D or A->B->D, fall back to self.merge_linear
            case 2: if both B and C are some op,
                    merged_cost = max(cost(B), cost(C))
        '''
        cnt = 0
        for sub_idx in range(self.total_rank):
            cnt += self._merge_branch(
                self.nodes[sub_idx], self.runtime_graph[sub_idx], is_bwd=False
            )
            cnt += self._merge_branch(
                self.nodes[sub_idx], self.runtime_graph[sub_idx], is_bwd=True
            )
        return cnt

    def _merge_linear(self, nodes, runtime_graph, is_bwd=False):
        reduct_cnt = 0
        rt_nodes_id = list(runtime_graph.keys())
        for node_id in rt_nodes_id:
            if node_id not in runtime_graph.keys():
                continue
            node = nodes[node_id]
            if not is_bwd == node.is_bwd or node.is_optim:
                continue
            edges = runtime_graph[node_id]
            ind = len(edges[PRED])  # in_degree
            if ind == 1:  # only depend on one node
                pred_id = edges[PRED][0]
                pred = nodes[pred_id]
                merged_node_id, merged_node = self._merge_node(
                    [node, pred], merge_type='linear', nodes=nodes
                )
                nodes[merged_node_id] = merged_node
                runtime_graph[merged_node_id] = [[], []]

                # delete edges and add new edges
                succ = None
                try:
                    runtime_graph[merged_node_id][SUCC] = copy.deepcopy(
                        edges[SUCC]
                    )

                    if len(runtime_graph[pred_id][SUCC]) > 1:
                        # predecessor has more than 1 successor
                        # the merged_node is to inherit the rest of its successors
                        succ = runtime_graph[pred_id][SUCC]
                        succ.remove(node_id)
                        runtime_graph[merged_node_id][SUCC] += succ
                    runtime_graph[merged_node_id][PRED] = runtime_graph[
                        pred_id
                    ][PRED]
                except:
                    pass
                try:
                    for i in runtime_graph[pred_id][PRED]:
                        try:
                            runtime_graph[i][SUCC].remove(pred_id)
                        except:
                            continue
                        runtime_graph[i][SUCC].append(merged_node_id)
                except:
                    pass

                try:
                    for i in edges[SUCC]:
                        runtime_graph[i][PRED].remove(node_id)
                        runtime_graph[i][PRED].append(merged_node_id)
                except:
                    pass
                if succ is not None:
                    for i in succ:
                        try:
                            runtime_graph[i][PRED].remove(pred_id)
                        except:
                            continue
                        runtime_graph[i][PRED].append(merged_node_id)

                runtime_graph.pop(node_id)
                try:
                    runtime_graph.pop(pred_id)
                except:
                    continue
                reduct_cnt += 1
                self.eliminate_multi_edges(runtime_graph)
                break
        return reduct_cnt  # the number of nodes that have been reduced

    def _merge_branch(self, nodes, runtime_graph, is_bwd=False):
        reduct_cnt = 0
        rt_nodes_id = list(runtime_graph.keys())
        for node_id in rt_nodes_id:
            node = nodes[node_id]
            if not is_bwd == node.is_bwd or node.is_optim:
                continue
            edges = runtime_graph[node_id]
            outd = len(edges[SUCC])  # out_degree
            if outd > 1:  # branch out
                succ_nodes_id = edges[SUCC]

                succ_to_elim = []
                for succ_id in succ_nodes_id:
                    for succ_2_id in succ_nodes_id:
                        try:
                            tmp = runtime_graph[succ_2_id][SUCC]
                        except:
                            continue
                        if succ_id in tmp:
                            succ_to_elim.append(succ_id)
                            break
                for id in succ_to_elim:
                    edges[SUCC].remove(id)
                    runtime_graph[id][PRED].remove(node_id)
                    reduct_cnt += 1

                to_merge = True
                try:
                    if (
                        len(edges[SUCC]) < 1
                        or len(runtime_graph[edges[SUCC][0]][SUCC]) < 1
                    ):
                        continue
                except:
                    continue
                end_node_id = runtime_graph[edges[SUCC][0]][SUCC][0]
                for i in succ_nodes_id:
                    try:
                        if (
                            len(runtime_graph[i][SUCC]) != 1
                            or runtime_graph[i][SUCC][0] != end_node_id
                        ):
                            to_merge = False  # if branches has different end node, we don't merge them
                            break
                    except:
                        continue
                if to_merge and len(succ_nodes_id) > 1:
                    to_merge_node_list = [nodes[i] for i in succ_nodes_id]
                    merged_node_id, merged_node = self._merge_node(
                        to_merge_node_list, merge_type='branch', nodes=nodes
                    )
                    nodes[merged_node_id] = merged_node
                    runtime_graph[merged_node_id] = [[], []]

                    # delete edges and add new edges
                    runtime_graph[merged_node_id][SUCC] = [end_node_id]
                    runtime_graph[merged_node_id][PRED] = edges[PRED]

                    runtime_graph[end_node_id][PRED] = [merged_node_id]
                    runtime_graph[node_id][SUCC] = [merged_node_id]

                    try:
                        for i in succ_nodes_id:
                            runtime_graph.pop(i)
                        reduct_cnt += len(to_merge_node_list) - 1
                        break
                    except:
                        pass
        return reduct_cnt

    def get_runtime_cost(self):
        def get_node_cost(node):
            node_cost = node.cost + self.opcall_overhead
            if isinstance(node, MergedOpsCostNode):
                for it in node.node_list:
                    node_cost += self.opcall_overhead
            return node_cost

        for sub_idx in range(self.total_rank):
            fwd_cost = 0
            bwd_cost = 0
            optim_cost = 0
            for node_id in self.runtime_graph[sub_idx].keys():
                node = self.nodes[sub_idx][node_id]
                if node.is_optim:
                    optim_cost += get_node_cost(node)
                elif node.is_bwd:
                    bwd_cost += get_node_cost(node)
                else:
                    fwd_cost += get_node_cost(node)
            self.fwd_time.append(fwd_cost)
            self.bwd_time.append(bwd_cost)
            self.optim_time.append(optim_cost)
        return self.fwd_time, self.bwd_time, self.optim_time

    def get_mem(self):
        static_list = []
        top_list = []
        for sub_idx in range(self.total_rank):
            static_mem, cur_mem, top_mem = self._simulate_mem(
                self.nodes[sub_idx], self.origin_graph[sub_idx]
            )
            static_list.append(static_mem)
            top_list.append(top_mem)
        return static_list, top_list

    def _simulate_mem(self, nodes, origin_graph):
        q = queue.Queue(1024)
        sim_graph = copy.deepcopy(origin_graph)
        for node_id, node in nodes.items():
            if len(sim_graph[node_id][PRED]) == 0:
                q.put(node_id)

        q.put('nop')
        cur_mem = 0
        top_mem = -1
        static_mem = 0
        while not q.empty():
            node_id = q.get()
            node = None
            size = 0
            if node_id == 'nop':
                top_mem = max(cur_mem, top_mem)
                if q.empty():
                    break
                else:
                    q.put(node_id)
                    continue
            else:
                node = nodes[node_id]
            if node.type == CostNodeType.VARIABLE:
                size = node.get_size()
                if node.node.persistable:
                    static_mem += size
                cur_mem += size
            edges = sim_graph[node_id]
            if not (
                node.type == CostNodeType.VARIABLE and node.node.persistable
            ):
                for succ_id in edges[SUCC]:
                    sim_graph[succ_id][PRED].remove(node_id)
                    if len(sim_graph[succ_id][PRED]) == 0:
                        q.put(succ_id)
            for pred_id in edges[PRED]:
                pred = nodes
                if pred.type == CostNodeType.VARIABLE:
                    sim_graph[pred_id][SUCC].remove(node_id)
                    if (
                        len(sim_graph[pred_id][SUCC]) == 0
                        and not pred.node.persistable
                    ):
                        cur_mem -= pred.get_size()
        return static_mem, cur_mem, top_mem

    def get_pipeline_time(self):
        if self.pp2rank is None:
            return self.fwd_time[0] + self.bwd_time[0] + self.optim_time[0]
        else:
            return self._simulate_pipeline()

    def _simulate_pipeline(self):
        stage_num = len(self.pp2rank)
        event_list = []
        global_time = [0] * stage_num
        total_time = 0
        fwd_cnt = list(range(stage_num, 0, -1))
        bwd_cnt = [self.microbatch_num] * stage_num
        q = queue.Queue(1024)

        for i in range(self.microbatch_num):
            q.put(PipeEvent(0, 'fwd', self.fwd_time[0]))

        while not q.empty():
            e = q.get()
            stid = e.stage_id
            if e.name == 'fwd':
                if fwd_cnt[stid] > 0:
                    e.s_time = max(global_time[stid], e.s_time)
                    e.e_time = e.s_time + e.duration
                    event_list.append(e)
                    if stid != stage_num - 1:
                        q.put(
                            PipeEvent(
                                stid + 1,
                                'fwd',
                                self.fwd_time[stid + 1],
                                start_time=e.e_time,
                            )
                        )
                    else:
                        q.put(
                            PipeEvent(
                                stid,
                                'bwd',
                                self.bwd_time[stid],
                                start_time=e.e_time,
                            )
                        )
                    fwd_cnt[stid] -= 1
                    global_time[stid] = e.e_time
                else:
                    q.put(e)
            elif e.name == 'bwd':
                e.s_time = max(global_time[stid], e.s_time)
                e.e_time = e.s_time + e.duration
                event_list.append(e)
                if stid != 0:
                    q.put(
                        PipeEvent(
                            stid - 1,
                            'bwd',
                            self.bwd_time[stid - 1],
                            start_time=e.e_time,
                        )
                    )
                fwd_cnt[stid] += 1
                bwd_cnt[stid] -= 1
                if bwd_cnt[stid] == 0:
                    q.put(
                        PipeEvent(
                            stid,
                            'optim',
                            self.optim_time[stid],
                            start_time=e.e_time,
                        )
                    )
                global_time[stid] = e.e_time
            elif e.name == 'optim':
                e.s_time = max(global_time[stid], e.s_time)
                e.e_time = e.s_time + e.duration
                event_list.append(e)
                global_time[stid] = e.e_time
            else:
                raise NotImplementedError(
                    f'This type of pipe event is not supported yet.{e.name}'
                )

        for t in global_time:
            total_time = max(total_time, t)
        return total_time

    def get_cost(self):
        cost = Cost()
        static_mem, peak_mem = self.get_mem()
        cost.static_mem = static_mem
        cost.peak_mem = peak_mem
        self.merge_comm()
        while True:
            cnt = 0
            cnt += self.merge_linear()
            cnt += self.merge_branch()
            if cnt == 0:  # can't be further merged
                break
        self.get_runtime_cost()
        cost.runtime = self.get_pipeline_time()
        return cost

    def init(self, distributed_program):
        self.parse_program(distributed_program)
        self.build_op_graph()
        for sub_idx in range(self.total_rank):
            self.eliminate_multi_edges(self.op_graph[sub_idx])
        self.build_runtime_graph()


def estimate_cost(
    distributed_program,
    cluster,
    pipeline_config,
    standalone_cost_data,
    batch_size,
):
    """
    Estimated cost from distributed program, cluster model and distributed settings.

    Args:
        distributed_program(list): list of paddle programs
        cluster(Cluster): cluster model
        standalone_cost_data(CostData): cost data given by paddle.core
        batch_size(int): batch size of the training workload
        pipeline_config(list): configuration of pipeline stage allocation
    """
    # the following line is left for now, cluster model will be involved in the future
    assert cluster is None, "For now, cluster remains None"
    cm_ctx = CostModel(
        cluster=cluster,
        batch_size=batch_size,
        standalone_cost_data=standalone_cost_data,
        pipeline_config=pipeline_config,
    )
    cm_ctx.init(distributed_program)
    cost = cm_ctx.get_cost()
    return cost
