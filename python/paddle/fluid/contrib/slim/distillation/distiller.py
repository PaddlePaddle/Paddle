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

__all__ = ['FSPDistiller', 'L2Distiller']


class L2Distiller(object):
    """
    Combine two layers from student net and teacher net by l2-loss.
    And add the loss into the total loss using for distillation training.
    """

    def __init__(self,
                 student_feature_map,
                 teacher_feature_map,
                 distillation_loss_weight=1):
        """
        Args:
            student_feature_map(str): The name of feature map from student network.
            teacher_feature_map(str): The name of feature map from teacher network.
                                      It's shape should be the same with student network.
            distillation_loss_weight(float): The weight of the l2-loss.
        """
        self.student_feature_map = student_feature_map
        self.teacher_feature_map = teacher_feature_map
        self.distillation_loss_weight = distillation_loss_weight

    def distiller_loss(self, graph):
        """
        Modify graph inplace to add l2-loss.
        Args: 
            graph(GraphWrapper): The graph to be modified.
        Returns:
            GraphWrapper: The modified graph.
        """
        distiller_pass = L2DistillerPass(self.student_feature_map,
                                         self.teacher_feature_map,
                                         self.distillation_loss_weight)
        dis_graph = distiller_pass.apply(graph)
        return dis_graph


class L2DistillerPass(object):
    """
    The pass used to add l2-loss.
    """

    def __init__(self,
                 student_feature_map,
                 teacher_feature_map,
                 distillation_loss_weight=1):
        """
        Args:
            student_feature_map(str): The name of feature map from student network.
            teacher_feature_map(str): The name of feature map from teacher network.
                                      It's shape should be the same with student network.
            distillation_loss_weight(float): The weight of the l2-loss.
        """
        self.student_feature_map = student_feature_map
        self.teacher_feature_map = teacher_feature_map
        self.distillation_loss_weight = distillation_loss_weight

    def apply(self, graph):
        ret_graph = graph
        with program_guard(ret_graph.program):

            student_feature_map = ret_graph.var(self.student_feature_map)._var
            teacher_feature_map = ret_graph.var(self.teacher_feature_map)._var
            l2loss = layers.reduce_mean(
                layers.square(student_feature_map - teacher_feature_map))

            distillation_loss = l2loss * self.distillation_loss_weight
            student_loss = ret_graph.var(ret_graph.out_nodes['loss'])._var
            loss = distillation_loss + student_loss

            ret_graph.out_nodes[
                'l2loss_' + self.student_feature_map + "_" +
                self.teacher_feature_map] = distillation_loss.name
            ret_graph.out_nodes['loss'] = loss.name
        return ret_graph


class FSPDistiller(object):
    """
    Combine layers from student net and teacher net by fsp-loss.
    """

    def __init__(self, student_pairs, teacher_pairs,
                 distillation_loss_weight=1):
        """
        Args:
            student_pairs(list<tuple>): Each tuple, with two variable names, in student_pairs indicates
                                        a section in student network. The variables in a tuple should
                                        have the same feature map size.
            teacher_pairs(list<tuple>): Each tuple, with two variable names, in teacher_pairs indicates
                                        a section in teacher network. The variables in a tuple should
                                        have the same feature map size. Varibale named teacher_pairs[i][j]
                                        should has the save channel number with that of variable named 
                                        student_pairs[i][j].

            distillation_loss_weight(float): The weight of the fsp-loss. default: 1.
        """
        self.student_pairs = student_pairs
        self.teacher_pairs = teacher_pairs
        self.distillation_loss_weight = distillation_loss_weight

    def distiller_loss(self, graph):
        """
        Modify graph inplace to add fsp-loss.
        Args: 
            graph(GraphWrapper): The graph to be modified.
        Returns:
            GraphWrapper: The modified graph.
        """
        distiller_pass = FSPDistillerPass(self.student_pairs,
                                          self.teacher_pairs,
                                          self.distillation_loss_weight)
        dis_graph = distiller_pass.apply(graph)
        return dis_graph


class FSPDistillerPass(object):
    '''
    Combine layers from student net and teacher net by fsp-loss.
    '''

    def __init__(self, s_pairs, t_pairs, distillation_loss_weight=1):
        """
        Args:
            s_pairs(list<tuple>): Each tuple, with two variable names, in student_pairs indicates
                                        a section in student network. The variables in a tuple should
                                        have the same feature map size.
            t_pairs(list<tuple>): Each tuple, with two variable names, in teacher_pairs indicates
                                        a section in teacher network. The variables in a tuple should
                                        have the same feature map size. Varibale named teacher_pairs[i][j]
                                        should has the save channel number with that of variable named 
                                        student_pairs[i][j].

            distillation_loss_weight(float): The weight of the fsp-loss. default: 1.
        """
        self.s_pairs = s_pairs
        self.t_pairs = t_pairs
        self.distillation_loss_weight = distillation_loss_weight

    def apply(self, graph):
        ret_graph = graph
        with program_guard(ret_graph.program):
            losses = []
            for s_pair, t_pair in zip(self.s_pairs, self.t_pairs):
                s_pair_start = ret_graph.var(s_pair[0])._var
                s_pair_end = ret_graph.var(s_pair[1])._var
                s_fsp_matrix = self._fsp_matrix(s_pair_start, s_pair_end)
                t_pair_start = ret_graph.var(t_pair[0])._var
                t_pair_end = ret_graph.var(t_pair[1])._var
                t_fsp_matrix = self._fsp_matrix(t_pair_start, t_pair_end)
                l2_loss = layers.reduce_mean(
                    layers.square(s_fsp_matrix - t_fsp_matrix))
                losses.append(l2_loss)
            distillation_loss = layers.sum(
                losses) * self.distillation_loss_weight
            student_loss = ret_graph.var(ret_graph.out_nodes['loss'])._var
            loss = distillation_loss + student_loss

            ret_graph.out_nodes[
                'fsp_distillation_loss'] = distillation_loss.name
            ret_graph.out_nodes['loss'] = loss.name
        return ret_graph

    def _fsp_matrix(self, fea_map_0, fea_map_1):
        return layers.fsp_matrix(fea_map_0, fea_map_1)
