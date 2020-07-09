#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet as fleet

def sum(input, scope=None):
    """
    distributed sum in fleet

    Args:
        input(Variable): output of a layer
        scope(Scope): specific scope, default is None

    Returns:
        global_metric(numpy.array): sum array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_add(, global_cnt)
          fluid.layers.assign(tmp, global_cnt)
          
          # in train.py, after train or infer
          print("sum array: ", paddle.fleet.sum(global_cnt))
    """
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    input = np.array(scope.find_var(input.name).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    fleet._role_maker._all_reduce(input, output, mode="sum")
    output = output.reshape(old_shape)
    return output

def max(input, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    input = np.array(scope.find_var(input.name).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    fleet._role_maker._all_reduce(input, output, mode="max")
    output = output.reshape(old_shape)
    return output

def min(input, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    input = np.array(scope.find_var(input.name).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    fleet._role_maker._all_reduce(input, output, mode="max")
    output = output.reshape(old_shape)
    return output

def auc(stat_pos, stat_neg, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    # auc pos bucket
    pos = np.array(scope.find_var(stat_pos.name).get_tensor())
    # auc pos bucket shape
    old_pos_shape = np.array(pos.shape)
    # reshape to one dim
    pos = pos.reshape(-1)
    global_pos = np.copy(pos) * 0
    # mpi allreduce
    fleet._role_maker._all_reduce(pos, global_pos)
    # reshape to its original shape
    global_pos = global_pos.reshape(old_pos_shape)

    # auc neg bucket
    neg = np.array(scope.find_var(stat_neg.name).get_tensor())
    old_neg_shape = np.array(neg.shape)
    neg = neg.reshape(-1)
    global_neg = np.copy(neg) * 0
    fleet._role_maker._all_reduce(neg, global_neg)
    global_neg = global_neg.reshape(old_neg_shape)

    # calculate auc
    num_bucket = len(global_pos[0])
    area = 0.0
    pos = 0.0
    neg = 0.0
    new_pos = 0.0
    new_neg = 0.0
    total_ins_num = 0
    for i in xrange(num_bucket):
        index = num_bucket - 1 - i
        new_pos = pos + global_pos[0][index]
        total_ins_num += global_pos[0][index]
        new_neg = neg + global_neg[0][index]
        total_ins_num += global_neg[0][index]
        area += (new_neg - neg) * (pos + new_pos) / 2
        pos = new_pos
        neg = new_neg

    auc_value = None
    if pos * neg == 0 or total_ins_num == 0:
        auc_value = 0.5
    else:
        auc_value = area / (pos * neg)

    fleet._role_maker._barrier_worker()
    return auc_value

def mae(abserr, total_ins_num, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    metric = np.array(scope.find_var(abserr.name).get_tensor())
    old_metric_shape = np.array(metric.shape)
    metric = metric.reshape(-1)
    global_metric = np.copy(metric) * 0
    fleet._role_maker._all_reduce(metric, global_metric)
    global_metric = global_metric.reshape(old_metric_shape)
    mae_value = global_abserr[0] / total_ins_num
    return mae_value

def rmse(sqrerr, total_ins_num, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    metric = np.array(scope.find_var(sqrerr.name).get_tensor())
    old_metric_shape = np.array(metric.shape)
    metric = metric.reshape(-1)
    global_metric = np.copy(metric) * 0
    fleet._role_maker._all_reduce(metric, global_metric)
    global_metric = global_metric.reshape(old_metric_shape)
    rmse_value = math.sqrt(global_abserr[0] / total_ins_num)
    return rmse_value

def mse(sqrerr, total_ins_num, scope):
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    metric = np.array(scope.find_var(sqrerr.name).get_tensor())
    old_metric_shape = np.array(metric.shape)
    metric = metric.reshape(-1)
    global_metric = np.copy(metric) * 0
    fleet._role_maker._all_reduce(metric, global_metric)
    global_metric = global_metric.reshape(old_metric_shape)
    mse_value = global_abserr[0] / total_ins_num
    return mse_value

def acc(correct, total):
    
    

