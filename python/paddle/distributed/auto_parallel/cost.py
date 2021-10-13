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
import numpy as np
import json
import queue
import copy
from enum import Enum
import paddle

SUCC = 0  # successor
PRED = 1  # predecessor


class NodeType(Enum):
    DEFAULT = 0
    COMPUTATION = 1
    COMMUNICATION = 2
    VARIABLE = 3
    MERGED = 4
    NOP = 5


class Cost(object):
    def __init__(self):
        self.runtime = None
        self.static_mem = None
        self.peak_mem = None


class CostModelMode(Enum):
    DEFAULT = 0
    BENCHMARKING = 1  # costs based on trial runs
    ANALYSIS = 2  # costs based on analysis
    MIXED = 3


class NodeBase(object):
    def __init__(self, node, node_type, id=None):
        self.id = id
        self.node = node
        self.type = node_type
        self.cost = 0

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost


class MergedNode(NodeBase):
    def __init__(self, node_type, id=None, base_node_list=None):
        super(MergedNode, self).__init__(None, node_type, id)
        self.node_list = base_node_list


class CommNode(NodeBase):
    def __init__(self, node, node_type, id=None, comm_node_list=None):
        super(CommNode, self).__init__(None, node_type, id)
        self.node_list = comm_node_list
        self.ranks = []
        self.comm_type = node.type

    def set_ranks(self, ranks):
        self.ranks = ranks

    def set_shapes(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def set_cost(self, cluster=None):
        # ref: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
        # should get from `cluster`
        BANDWIDTH = 32 * 1024 / 1000  # MB/ms, V100 PCIe
        num_ranks = len(self.ranks)
        comm_volumn = np.prod(self.input_shape) * 4

        if 'allreduce' in self.comm_type:
            self.cost = comm_volumn / (BANDWIDTH * num_ranks /
                                       (2 * (num_ranks - 1)))
        elif 'gather' in self.comm_type:
            self.cost = comm_volumn / (BANDWIDTH * num_ranks / (num_ranks - 1))
        elif 'broadcast' in self.comm_type:
            self.cost = comm_volumn / BANDWIDTH
        elif 'send' in self.comm_type or 'recv' in self.comm_type:
            self.cost = comm_volumn / BANDWIDTH
        else:
            self.cost = 0

    def get_cost(self, cluster=None):
        return self.cost


class VarNode(NodeBase):
    def __init__(self,
                 node,
                 node_type,
                 id=None,
                 base_node_list=None,
                 batch_size=None,
                 shared_node_id=None):
        super(VarNode, self).__init__(node, node_type, id)
        self.shape = node.shape
        self.dtype = node.dtype
        self.dtype_factor = 1
        self.persistable = None
        self.shared_node_id = shared_node_id
        self.cost = 0.0
        if self.dtype == paddle.float32 or node.dtype == paddle.int32:
            self.dtype_factor *= 4
        elif node.dtype == paddle.int64:
            self.dtype_factor *= 8
        else:
            raise NotImplementedError("{} not counted".format(v.node.dtype))

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


class CompNode(NodeBase):
    def __init__(self, node, node_type, id=None):
        super(CompNode, self).__init__(node, node_type, id)

    def set_cost(self, cost_data):
        op_name = self.node.type
        if op_name in cost_data.keys():
            self.cost = cost_data[op_name]
        else:
            self.cost = 0.0
        return self.cost

    def get_cost(self):
        return self.cost


class PipeEvent(object):
    def __init__(self, stage_id, event_name, duration, start_time=-1):
        self.stage_id = stage_id
        self.name = event_name
        self.duration = duration
        self.s_time = start_time
        self.e_time = -1


class CostModelContext(object):
    def __init__(self,
                 mode=CostModelMode.BENCHMARKING,
                 cluster=None,
                 batch_size=1,
                 microbatch_num=1,
                 opcall_overhead=0,
                 bwd_ratio=1.5,
                 update_time=0,
                 single_cost_data=None,
                 process_mesh=None):
        self.mode = mode

        # parameters
        self.opcall_overhead = opcall_overhead
        self.bwd_ratio = bwd_ratio
        self.batch_size = batch_size
        self.update_time = update_time
        self.microbatch_num = microbatch_num

        self.nodes = {}  # name -> node

        self.origin_graph = {}  # original graph
        self.op_graph = {}  # op graph (no variables nodes)
        self.rt_graph = {}  # runtime graph, for simulation

        self.cluster = cluster
        self.cost_data = single_cost_data
        self.process_mesh = process_mesh

        self.rank2dim = {}
        self.dp_group = {}
        self.mp_group = {}
        self.pp_send_peer = {}
        self.pp_recv_peer = {}
        for pp_id in range(self.process_mesh.topology[0]):
            for dp_id in range(self.process_mesh.topology[1]):
                for mp_id in range(self.process_mesh.topology[2]):
                    idx = mp_id + dp_id * self.process_mesh.topology[2] + \
                            pp_id * self.process_mesh.topology[2] * self.process_mesh.topology[1]
                    rank = self.process_mesh.process_group[idx]
                    self.rank2dim[rank] = [pp_id, dp_id, mp_id]

        self.fwd_time = []
        self.bwd_time = []
        self.optim_time = []

    def _parse_sub_program(self, program, nodes, graph, cost_data):
        assert len(
            program.blocks) == 1, "Program more than 1 block not supported."
        block = program.blocks[0]

        for var in block.vars.values():
            var_id = var.name
            nodes[var_id] = VarNode(var, NodeType.VARIABLE, var_id)
            graph[var_id] = [[], []]

        for op in block.ops:
            op_id = op.type + "_" + str(op.idx)
            if op.type.startswith('c_') or op.type.startswith(
                    'send') or op.type.startswith('recv'):
                op_node = CommNode(op, NodeType.COMMUNICATION, op_id)
            else:
                op_node = CompNode(op, NodeType.COMPUTATION, op_id)
                op_node.set_cost(cost_data)

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
            if op_node.type == NodeType.COMMUNICATION:
                op_node.set_shapes(comm_input_shape, comm_output_shape)

        # resolve hazard: rename the r/w hazard variable nodes to ensure self.origin_graph is a DAG
        new_var_dict = {}
        for node_id, node in nodes.items():
            if node.type == NodeType.VARIABLE and node.node.persistable:
                write_op_cnt = 0
                for pred_id in graph[node_id][PRED]:
                    pred = nodes[pred_id]
                    if pred.type == NodeType.COMPUTATION and (
                            pred_id in graph[node_id][SUCC]):

                        graph[pred_id][SUCC].remove(node_id)
                        graph[node_id][PRED].remove(pred_id)

                        write_op_cnt += 1
                        new_var_id = node_id + '_write_{}'.format(write_op_cnt)
                        new_var = VarNode(
                            node.node,
                            NodeType.VARIABLE,
                            new_var_id,
                            shared_node_id=node_id)

                        graph[new_var_id] = [[], []]
                        graph[pred_id][SUCC].append(new_var_id)
                        graph[new_var_id][PRED].append(pred_id)

                        new_var_dict[new_var_id] = new_var
        for k, v in new_var_dict.items():
            nodes[k] = v
        return nodes

    def parse_program(self, distributed_program):
        self.distributed_program = distributed_program
        sub_prog_cnt = len(distributed_program)
        self.nodes = [] * sub_prog_cnt
        self.origin_graph = [] * sub_prog_cnt  # original graph
        self.op_graph = [] * sub_prog_cnt  # op graph (no variables nodes)
        self.rt_graph = [] * sub_prog_cnt  # runtime graph, for simulation

        for sub_idx, sub_prog in enumerate(distributed_program):
            self.nodes.append({})
            self.origin_graph.append({})
            self.op_graph.append({})
            self.rt_graph.append({})
            self._parse_sub_program(sub_prog, self.nodes[sub_idx],
                                    self.origin_graph[sub_idx],
                                    self.cost_data[self.rank2dim[sub_idx][0]])
        return self.nodes

    def _find_succ_op(self, node_id, sub_idx=0):
        succ_ops_id = []
        for succ_id in self.origin_graph[sub_idx][node_id][SUCC]:
            succ = self.nodes[sub_idx][succ_id]
            if succ.type == NodeType.COMMUNICATION or \
                succ.type == NodeType.COMPUTATION:
                succ_ops_id.append(succ_id)
            elif succ.type == NodeType.VARIABLE:
                succ_ops_id = succ_ops_id + self._find_succ_op(succ_id, sub_idx)
            else:
                raise NotImplementedError(
                    'This type of node not supported yet:{}'.format(succ.type))
        return succ_ops_id

    def build_op_graph(self):
        for sub_idx in range(len(self.distributed_program)):
            op_nodes_id = []
            for node_id, node in self.nodes[sub_idx].items():
                if node.type == NodeType.VARIABLE:
                    continue
                self.op_graph[sub_idx][node_id] = [[], []]
                op_nodes_id.append(node_id)
            for op_id in op_nodes_id:
                succ_nodes_id = self._find_succ_op(op_id, sub_idx)

                self.op_graph[sub_idx][op_id][SUCC] = succ_nodes_id
                for succ_id in succ_nodes_id:
                    self.op_graph[sub_idx][succ_id][PRED].append(op_id)

    def build_rt_graph(self):
        self.rt_graph = copy.deepcopy(self.op_graph)

    def elim_multi_edges(self, graph=None):
        for node_id, edges in graph.items():
            graph[node_id][PRED] = list(set(edges[PRED]))
            graph[node_id][SUCC] = list(set(edges[SUCC]))

    def merge_comm(self):
        for sub_idx in range(len(self.distributed_program)):
            dim = self.rank2dim[sub_idx]
            start_idx = dim[1] * self.process_mesh.topology[2] + \
                        dim[0] * self.process_mesh.topology[2] * self.process_mesh.topology[1]
            end_idx = (dim[1]+1) * self.process_mesh.topology[2] + \
                       dim[0] * self.process_mesh.topology[2] * self.process_mesh.topology[1]
            mp_group = list(
                map(lambda x: self.process_mesh.process_group[x],
                    range(start_idx, end_idx)))

            start_idx = dim[2] + dim[0] * self.process_mesh.topology[
                2] * self.process_mesh.topology[1]
            end_idx = start_idx + self.process_mesh.topology[
                2] * self.process_mesh.topology[1]
            dp_group = list(
                map(lambda x: self.process_mesh.process_group[x],
                    range(start_idx, end_idx, self.process_mesh.topology[2])))

            rank = self.process_mesh.process_group.index(sub_idx)
            if dim[0] != 0:  # first stage
                pp_pred_peer = self.process_mesh.process_group[
                    rank - self.process_mesh.topology[
                        2] * self.process_mesh.topology[1]]

            if dim[0] != (self.process_mesh.topology[0] - 1):  # last stage
                pp_succ_peer = self.process_mesh.process_group[
                    rank + self.process_mesh.topology[
                        2] * self.process_mesh.topology[1]]

            for node_id, edges in self.op_graph[sub_idx].items():
                node = self.nodes[sub_idx][node_id]
                if node_id.startswith('c_'):
                    if "@GRAD" in self.origin_graph[sub_idx][node_id][SUCC][0]:
                        node.set_ranks(dp_group)
                    else:
                        node.set_ranks(mp_group)
                    node.set_cost(self.cluster)

                elif node_id.startswith('send'):
                    if "@GRAD" in self.origin_graph[sub_idx][node_id][PRED][0]:
                        node.set_ranks([sub_idx, pp_pred_peer])
                    else:
                        node.set_ranks([sub_idx, pp_succ_peer])
                    node.set_cost(self.cluster)

                elif node_id.startswith('recv'):
                    if "@GRAD" in self.origin_graph[sub_idx][node_id][SUCC][0]:
                        node.set_ranks([sub_idx, pp_succ_peer])
                    else:
                        node.set_ranks([sub_idx, pp_pred_peer])
                    node.set_cost(self.cluster)
                else:
                    pass  # Not communication op

    def _merge_node(self, to_merge_node_list, merge_type='linear', nodes=None):
        nodes_list = []
        node_cost = 0
        for node in to_merge_node_list:
            if isinstance(node, MergedNode):
                nodes_list += node.node_list
            else:
                nodes_list.append(node.id)

            if merge_type == 'linear':
                node_cost += node.cost
            elif merge_type == 'branch':
                node_cost = max(node_cost, node.cost)
            else:
                raise NotImplementedError(
                    'This type of merging is not supported:{}'.format(
                        merge_type))

        merged_node_id = 'merged_' + str(len(nodes))
        merged_node = MergedNode(
            NodeType.MERGED, id=merged_node_id, base_node_list=nodes_list)
        merged_node.set_cost(node_cost)

        return merged_node_id, merged_node

    def merge_linear(self):
        '''
        This method does the following: 
        If X depends on Y only, they must be run sequentially.
            [ e.g. A ->- C ->- D   D and E depends on C only.] 
            [      B ->-/ \->- E   C depends on A and B.     ]
        We merge X and Y into a new node and sum up their cost time.
        '''
        cnt = 0
        for sub_idx in range(len(self.distributed_program)):
            cnt += self._merge_linear(self.nodes[sub_idx],
                                      self.rt_graph[sub_idx])
        return cnt

    def merge_branch(self):
        '''
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
        for sub_idx in range(len(self.distributed_program)):
            cnt += self._merge_branch(self.nodes[sub_idx],
                                      self.rt_graph[sub_idx])
        return cnt

    def _merge_linear(self, nodes, rt_graph):
        reduct_cnt = 0
        rt_nodes_id = list(rt_graph.keys())
        for node_id in rt_nodes_id:
            if node_id not in rt_graph.keys():
                # the node has already been deleted because of merging
                continue
            edges = rt_graph[node_id]
            ind = len(edges[PRED])  # in_degree
            if ind == 1:  # only depend on one node
                pred_id = edges[PRED][0]
                node = nodes[node_id]
                pred = nodes[pred_id]
                merged_node_id, merged_node = self._merge_node(
                    [node, pred], merge_type='linear', nodes=nodes)
                nodes[merged_node_id] = merged_node
                rt_graph[merged_node_id] = [[], []]

                # delete edges and add new edges
                succ = None
                rt_graph[merged_node_id][SUCC] = copy.deepcopy(edges[SUCC])
                if len(rt_graph[pred_id][SUCC]) > 1:
                    # predecessor has more than 1 successor
                    # the merged_node is to inherit the rest of its successors
                    succ = rt_graph[pred_id][SUCC]
                    succ.remove(node_id)
                    rt_graph[merged_node_id][SUCC] += succ
                rt_graph[merged_node_id][PRED] = rt_graph[pred_id][PRED]
                for i in rt_graph[pred_id][PRED]:
                    rt_graph[i][SUCC].remove(pred_id)
                    rt_graph[i][SUCC].append(merged_node_id)

                for i in edges[SUCC]:
                    rt_graph[i][PRED].remove(node_id)
                    rt_graph[i][PRED].append(merged_node_id)
                if succ is not None:
                    for i in succ:
                        rt_graph[i][PRED].remove(pred_id)
                        rt_graph[i][PRED].append(merged_node_id)

                rt_graph.pop(node_id)
                rt_graph.pop(pred_id)
                reduct_cnt += 1
        self.elim_multi_edges(rt_graph)
        return reduct_cnt  # the number of nodes that have been reduced

    def _merge_branch(self, nodes, rt_graph):
        reduct_cnt = 0
        rt_nodes_id = list(rt_graph.keys())
        for node_id in rt_nodes_id:
            edges = rt_graph[node_id]
            outd = len(edges[SUCC])  # out_degree
            if outd > 1:  # branch out
                succ_nodes_id = edges[SUCC]

                succ_to_elim = []
                for succ_id in succ_nodes_id:
                    for succ_2_id in succ_nodes_id:
                        tmp = rt_graph[succ_2_id][SUCC]
                        if succ_id in tmp:
                            succ_to_elim.append(succ_id)
                            break
                for id in succ_to_elim:
                    edges[SUCC].remove(id)
                    rt_graph[id][PRED].remove(node_id)
                    reduct_cnt += 1

                to_merge = True
                # print(edges)
                if len(edges[SUCC]) < 1 or len(rt_graph[edges[SUCC][0]][
                        SUCC]) < 1:
                    continue
                end_node_id = rt_graph[edges[SUCC][0]][SUCC][0]
                for i in succ_nodes_id:
                    if len(rt_graph[i][SUCC]) > 1 or \
                        rt_graph[i][SUCC][0] != end_node_id:
                        to_merge = False  # if branches has different end node, we don't merge them
                        break
                if to_merge:
                    to_merge_node_list = [nodes[i] for i in succ_nodes_id]
                    merged_node_id, merged_node = self._merge_node(
                        to_merge_node_list, merge_type='branch', nodes=nodes)
                    nodes[merged_node_id] = merged_node
                    rt_graph[merged_node_id] = [[], []]

                    # delete edges and add new edges
                    rt_graph[merged_node_id][SUCC] = [end_node_id]
                    rt_graph[merged_node_id][PRED] = edges[PRED]

                    rt_graph[end_node_id][PRED] = [merged_node_id]
                    rt_graph[node_id][SUCC] = [merged_node_id]

                    for i in succ_nodes_id:
                        rt_graph.pop(i)
                    reduct_cnt += len(to_merge_node_list) - 1
        return reduct_cnt

    def get_rt_cost(self):
        self.time_list = []
        for sub_idx in range(len(self.distributed_program)):
            time_cost = 0.0
            for node_id in self.rt_graph[sub_idx].keys():
                node = self.nodes[sub_idx][node_id]
                time_cost += node.get_cost()
                if isinstance(node, MergedNode):
                    for it in node.node_list:
                        time_cost += self.opcall_overhead
            self.time_list.append(time_cost)
        return self.time_list

    def get_mem(self):
        static_list = []
        top_list = []
        for sub_idx in range(len(self.distributed_program)):
            static_mem, cur_mem, top_mem = self._mem_sim(
                self.nodes[sub_idx], self.origin_graph[sub_idx])
            static_list.append(static_mem)
            top_list.append(top_mem)
        return static_list, top_list

    def _mem_sim(self, nodes, origin_graph):
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

            if node.type == NodeType.VARIABLE:
                size = node.get_size()
                if node.node.persistable:
                    static_mem += size
                cur_mem += size

            edges = sim_graph[node_id]
            if not (node.type == NodeType.VARIABLE and node.node.persistable):
                for succ_id in edges[SUCC]:
                    sim_graph[succ_id][PRED].remove(node_id)
                    if len(sim_graph[succ_id][PRED]) == 0:
                        q.put(succ_id)

            for pred_id in edges[PRED]:
                pred = nodes
                if pred.type == NodeType.VARIABLE:
                    sim_graph[pred_id][SUCC].remove(node_id)
                    if len(sim_graph[pred_id][
                            SUCC]) == 0 and not pred.node.persistable:
                        cur_mem -= pred.get_size()

        return static_mem, cur_mem, top_mem

    def get_pipeline_time(self):
        if len(self.time_list) <= 1:
            return self.time_list[0] + self.time_list[
                0] * self.bwd_ratio + self.update_time
        else:
            pp = list(
                map(lambda x: self.process_mesh.process_group[x],
                    range(0, self.process_mesh.topology[2] * self.process_mesh.
                          topology[1], self.process_mesh.topology[2])))

            self.fwd_time = [self.time_list[pp[i]] for i in range(len(pp))]
            self.bwd_time = [
                self.time_list[pp[i]] * self.bwd_ratio for i in range(len(pp))
            ]
            self.optim_time = [self.update_time for i in range(len(pp))]
        return self.sim_pipeline()

    def sim_pipeline(self):
        stage_num = self.process_mesh.topology[0]
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
                                start_time=e.e_time))
                    else:
                        q.put(
                            PipeEvent(
                                stid,
                                'bwd',
                                self.bwd_time[stid],
                                start_time=e.e_time))
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
                            start_time=e.e_time))
                fwd_cnt[stid] += 1
                bwd_cnt[stid] -= 1
                if bwd_cnt[stid] == 0:
                    q.put(
                        PipeEvent(
                            stid,
                            'optim',
                            self.optim_time[stid],
                            start_time=e.e_time))
                global_time[stid] = e.e_time
            elif e.name == 'optim':
                e.s_time = max(global_time[stid], e.s_time)
                e.e_time = e.s_time + e.duration
                event_list.append(e)
                global_time[stid] = e.e_time
            else:
                raise NotImplementedError(
                    'This type of pipe event is not supported yet.{}'.format(
                        e.name))

        for t in global_time:
            total_time = max(total_time, t)
        return total_time


def estimate_cost(distributed_program, cluster, process_mesh, single_cost_data,
                  batch_size):
    """
    Estimated cost from distributed program, cluster model and distributed settings.
    
    Args:
        distributed_program(list): list of paddle programs
        cluster(Cluster): cluster model 
        process_mesh(ProcessMesh): process mesh containing distributed settings
        single_cost_data(CostData): cost data given by paddle.core
        batch_size(int): batch size of the training workload 
    """
    # the following line is left for now, cluster model will be involved in the future
    assert cluster is None, "For now, cluster remains None"

    cost = Cost()
    cm_ctx = CostModelContext(
        cluster=cluster,
        batch_size=batch_size,
        single_cost_data=single_cost_data,
        process_mesh=process_mesh)
    cm_ctx.parse_program(distributed_program)
    cm_ctx.build_op_graph()
    for sub_idx in range(len(distributed_program)):
        cm_ctx.elim_multi_edges(cm_ctx.op_graph[sub_idx])
    cm_ctx.build_rt_graph()

    static_mem, peak_mem = cm_ctx.get_mem()
    cost.static_mem = static_mem
    cost.peak_mem = peak_mem

    cm_ctx.merge_comm()
    while True:
        cnt = 0
        cnt += cm_ctx.merge_linear()
        cnt += cm_ctx.merge_branch()
        if cnt == 0:  # can't be further merged
            break

    cost.runtime = cm_ctx.get_rt_cost()
    cost.runtime = cm_ctx.get_pipeline_time()

    return cost
