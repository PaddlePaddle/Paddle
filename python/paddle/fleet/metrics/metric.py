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
          tmp = fluid.layers.elementwise_add(cnt, global_cnt)
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

def max(input, scope=None):
    """
    distributed max in fleet

    Args:
        input(Variable): output of a layer
        scope(Scope): specific scope, default is None

    Returns:
        global_metric(numpy.array): max array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_max(cnt, global_cnt)
          fluid.layers.assign(tmp, global_cnt)

          # in train.py, after train or infer
          print("max array: ", paddle.fleet.max(global_cnt))
    """
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    input = np.array(scope.find_var(input.name).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    fleet._role_maker._all_reduce(input, output, mode="max")
    output = output.reshape(old_shape)
    return output

def min(input, scope=None):
    """
    distributed min in fleet

    Args:
        input(Variable): output of a layer
        scope(Scope): specific scope, default is None

    Returns:
        global_metric(numpy.array): min array

    Example:
        .. code-block:: python

          # in model.py
          input = fluid.layers.cast(some_input, dtype='float32')
          cnt = fluid.layers.reduce_sum(input)
          global_cnt = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp = fluid.layers.elementwise_min(cnt, global_cnt)
          fluid.layers.assign(tmp, global_cnt)

          # in train.py, after train or infer
          print("min array: ", paddle.fleet.min(global_cnt))
    """
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    input = np.array(scope.find_var(input.name).get_tensor())
    old_shape = np.array(input.shape)
    output = np.copy(input) * 0
    fleet._role_maker._all_reduce(input, output, mode="min")
    output = output.reshape(old_shape)
    return output

def auc(stat_pos, stat_neg, scope=None):
    """
    distributed auc in fleet

    Args:
        stat_pos(Variable): stat_pos in output of fluid.layers.auc
        stat_neg(Variable): stat_neg in output of fluid.layers.auc
        scope(Scope): specific scope, default is None

    Returns:
        auc_value(float): auc value

    Example:
        .. code-block:: python

          # in model.py
          similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(output, min=-15.0, max=15.0))
          binary_predict = fluid.layers.concat(
              input=[fluid.layers.elementwise_sub(fluid.layers.ceil(similarity_norm), similarity_norm), similarity_norm], axis=1)
          self.auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] =
              fluid.layers.auc(input=binary_predict, label=label, curve='ROC', num_thresholds=4096)

          # in train.py, after train or infer
          print("auc: ", paddle.fleet.auc(stat_pos, stat_neg))
    """
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

def mae(abserr, total_ins_num, scope=None):
    """
    distributed mae in fleet

    Args:
        abserr(Variable): abserr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(int|float): total train/infer instance count
        scope(Scope): specific scope, default is None

    Returns:
        mae(float): mae value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          print("mae: ", paddle.fleet.mae(abserr, total_ins_num))
    """

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

def rmse(sqrerr, total_ins_num, scope=None):
    """
    distributed rmse in fleet

    Args:
        sqrerr(Variable): sqrerr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(int|float): total train/infer instance count
        scope(Scope): specific scope, default is None

    Returns:
        rmse(float): rmse value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          print("rmse: ", paddle.fleet.rmse(sqrerr, total_ins_num))
    """
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

def mse(sqrerr, total_ins_num, scope=None):
    """
    distributed mse in fleet

    Args:
        sqrerr(Variable): sqrerr in output of fluid.contrib.layers.ctr_metric_bundle
        total_ins_num(int|float): total train/infer instance count
        scope(Scope): specific scope, default is None

    Returns:
        mse(float): mse value

    Example:
        .. code-block:: python

          # in model.py
          sqrerr, abserr, prob, q, pos, total = fluid.contrib.layers.ctr_metric_bundle(similarity_norm, fluid.layers.cast(x=label, dtype='float32'))

          # in train.py, after train or infer
          print("mse: ", paddle.fleet.mse(sqrerr, total_ins_num))
    """
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

def acc(correct, total, scope=None):
    """
    distributed accuracy in fleet

    Args:
        correct(Variable): correct Variable
        total(Variable): total Variable

    Returns:
        acc(float): accuracy value

    Example:
        .. code-block:: python

          # in model.py
          correct = fluid.layers.create_global_var(dtype='float32', shape=[1], value=0)
          total = fluid.layers.create_global_var(dtype='float32', shape=[1], value=0)
          acc = fluid.layers.acc(predict, label, k=1, correct=correct, total=total)

          global_correct = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp1 = fluid.layers.elementwise_min(correct, global_correct)
          fluid.layers.assign(tmp1, global_correct)

          global_total = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1], value=0)
          tmp2 = fluid.layers.elementwise_min(total, global_total)
          fluid.layers.assign(tmp2, global_total)

          # in train.py, after train or infer
          print("accuracy: ", paddle.fleet.acc(global_correct, global_total))
    """
    if scope is None:
        scope = fluid.global_scope()
    fleet._role_maker._barrier_worker()
    correct_num = np.array(scope.find_var(correct.name).get_tensor())
    total_num = np.array(scope.find_var(total.name).get_tensor())
    global_correct_num = np.copy(correct_num) * 0
    global_total_num = np.copy(total_num) * 0
    fleet._role_maker._all_reduce(correct_num, global_correct_num)
    fleet._role_maker._all_reduce(total_num, global_total_num)
    return float(global_correct_num[0]) / float(global_total_num)
    

