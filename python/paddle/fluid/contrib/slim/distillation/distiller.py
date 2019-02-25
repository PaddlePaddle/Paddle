# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .... import layers
from .... import optimizer
from .... import Executor
from .... import Program
from .... import program_guard
from .... import regularizer

__all__ = ['FSPDistiller']


class FSPDistiller(object):
    def __init__(self,
                 student_pairs=None,
                 teacher_pairs=None,
                 distillation_loss_weight=1):
        self.student_pairs = student_pairs
        self.teacher_pairs = teacher_pairs
        self.distillation_loss_weight = distillation_loss_weight

    def _feature_map_pairs(self, graph):
        pairs = []
        sizes = []
        pair = []
        pre_size = None
        for op in graph.all_ops():
            if op.type == 'conv2d':
                out_var_name = op.output('Output')[0]
                feature_map_size = graph.get_var(out_var_name).shape[2:]
                if feature_map_size != pre_size:
                    if len(pair) == 2 and pair[1] != None:
                        pairs.append(pair)
                        sizes.append(pre_size)
                    pair = [out_var_name, None]
                else:
                    pair[1] = out_var_name
                pre_size = feature_map_size
        if len(pair) == 2 and pair[1] != None:
            pairs.append(pair)
            sizes.append(pre_size)
        return pairs

    def distiller_graph(self, student, teachers, optimizer, place):
        """
        Generate distillation training graph.
        """
        teacher = teachers[0]
        for var in teacher.program.list_vars():
            var.stop_gradient = True
        # step 1: merge student and teacher into graph
        graph = student.clone()
        graph.merge(teacher)
        if not self.student_pairs:
            self.student_pairs = self._feature_map_pairs(student)
        if not self.teacher_pairs:
            self.teacher_pairs = self._feature_map_pairs(teacher)
        # step 2: add fsp loss and backward ops
        distiller_pass = FSPDistillerPass(self.student_pairs,
                                          self.teacher_pairs, optimizer, place,
                                          self.distillation_loss_weight)
        dis_graph = distiller_pass.apply(graph)
        return dis_graph


class FSPDistillerPass(object):
    '''
    Convert graph to fsp distillation training graph
    by adding fsp loss and backward operators.
    '''

    def __init__(self,
                 s_pairs,
                 t_pairs,
                 distiller_optimizer,
                 place,
                 distillation_loss_weight=1):
        self.s_pairs = s_pairs
        self.t_pairs = t_pairs
        self.distillation_loss_weight = distillation_loss_weight
        self.distiller_optimizer = distiller_optimizer
        self.place = place

    def apply(self, graph):
        #        ret_graph = graph.clone()
        ret_graph = graph
        startup_program = Program()
        with program_guard(ret_graph.program, startup_program):
            losses = []
            for s_pair, t_pair in zip(self.s_pairs, self.t_pairs):
                s_pair_start = ret_graph.get_var(s_pair[0])
                s_pair_end = ret_graph.get_var(s_pair[1])
                s_fsp_matrix = self._fsp_matrix(s_pair_start, s_pair_end)
                t_pair_start = ret_graph.get_var(t_pair[0])
                t_pair_end = ret_graph.get_var(t_pair[1])
                t_fsp_matrix = self._fsp_matrix(t_pair_start, t_pair_end)
                #                layers.Print(t_fsp_matrix, summarize=10)
                l2_loss = layers.reduce_mean(
                    layers.square(s_fsp_matrix - t_fsp_matrix))
                losses.append(l2_loss)
            distillation_loss = layers.sum(
                losses) * self.distillation_loss_weight
            student_loss = ret_graph.get_var(ret_graph.out_nodes['cost'])
            loss = distillation_loss + student_loss

            #            distiller_optimizer = optimizer.Momentum(
            #                momentum=0.9,
            #                learning_rate=layers.piecewise_decay(
            #                    boundaries=[66666, 66666 * 2],
            #                    values=[0.001, 0.0001, 0.00001]),
            #                regularization=regularizer.L2Decay(4e-5))

            self.distiller_optimizer.minimize(loss)

            exe = Executor(self.place)
            # init variable created when append backward ops. Such as leaning rate
            # and accumulators in some optimizer.
            exe.run(startup_program, scope=ret_graph.scope)
            ret_graph.out_nodes['distillation_loss'] = distillation_loss.name
            ret_graph.out_nodes['student_loss'] = student_loss.name
            ret_graph.out_nodes['cost'] = loss.name
        return ret_graph

    def _fsp_matrix(self, fea_map_0, fea_map_1):
        return layers.fsp_matrix(fea_map_0, fea_map_1)
