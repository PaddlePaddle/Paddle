#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import math

from ..layer_helper import LayerHelper, unique_name
from ..framework import Variable, default_startup_program
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant
import nn, ops


def _allreduce(x, out=None, reduce_type="sum", sync_mode=False):
    helper = LayerHelper("allreduce", **locals())
    # Convert string reduce type to op int type
    red_typ_int = 0
    if reduce_type == "sum":
        red_typ_int = 0
    elif reduce_type == "prod":
        red_typ_int = 1
    elif reduce_type == "max":
        red_typ_int = 2
    elif reduce_type == "min":
        red_typ_int = 3
    else:
        raise TypeError("reduce type can only be [sum|prod|max|min]")

    if out is None:
        out = helper.create_variable(
            name=unique_name.generate_with_ignorable_key(".".join(
                [x.name, 'tmp'])),
            shape=x.shape,
            dtype=x.dtype,
            type=x.type,
            persistable=x.persistable,
            stop_gradient=True)
    helper.append_op(
        type='allreduce',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={"reduce_type": red_typ_int,
               "sync_mode": sync_mode})
    return out


def _broadcast(x, root, sync_mode=False):
    helper = LayerHelper("broadcast", **locals())
    helper.append_op(
        type='broadcast',
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={"sync_mode": sync_mode,
               "root": root})
    return x


def _c_allreduce(x,
                 out=None,
                 reduce_type='sum',
                 ring_id=0,
                 use_calc_stream=False):
    helper = LayerHelper('c_allreduce', **locals())

    if reduce_type not in ['sum', 'prob', 'max', 'min']:
        raise TypeError('reduce type can only be "sum|prod|max|min]"')

    op_type = 'c_allreduce_' + reduce_type
    if out is None:
        out = helper.create_variable(
            name=unique_name.generate_with_ignorable_key('.'.join(
                [x.name, op_type])),
            shape=x.shape,
            dtype=x.dtype,
            type=x.type,
            persistable=x.persistable)

    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'ring_id': ring_id,
               'use_calc_stream': use_calc_stream})
    return out


def _c_broadcast(x, root=0, ring_id=0, use_calc_stream=False):
    op_type = 'c_broadcast'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={
            'root': root,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return x


def _c_allgather(x, nranks, ring_id=0, use_calc_stream=False):
    op_type = 'c_allgather'
    helper = LayerHelper(op_type, **locals())
    out_shape = list(x.shape[:])
    if out_shape[0] > 0:
        out_shape[0] *= nranks
    out = helper.create_variable(
        name=unique_name.generate_with_ignorable_key('.'.join(
            [x.name, op_type])),
        shape=out_shape,
        dtype=x.dtype,
        type=x.type,
        persistable=x.persistable)
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={
            'nranks': nranks,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return out


def _c_reducescatter(x, nranks, ring_id=0, use_calc_stream=False):
    if not isinstance(x, Variable):
        raise TypeError('x must be a Variable')

    if x.shape[0] % nranks != 0:
        raise ValueError('x.shape[0](%d) cannot be evenly divided by nranks(%d)'
                         % (x.shape[0], nranks))

    op_type = 'c_reducescatter'
    helper = LayerHelper(op_type, **locals())
    out_shape = list(x.shape[:])
    if out_shape[0] > 0:
        out_shape[0] //= nranks
    out = helper.create_variable(
        name=unique_name.generate_with_ignorable_key('.'.join(
            [x.name, op_type])),
        shape=out_shape,
        dtype=x.dtype,
        type=x.type,
        persistable=x.persistable)
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={
            'nranks': nranks,
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream
        })
    return out


def _c_sync_calc_stream(x):
    op_type = 'c_sync_calc_stream'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type, inputs={'X': [x]}, outputs={'Out': [x]})
    return x


def _c_sync_comm_stream(x, ring_id):
    op_type = 'c_sync_comm_stream'
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={'ring_id': ring_id})
    return x


class DistributedClassifier(object):
    '''
    Tookit for distributed classification, in which the parameter of the last
    full-connected layer is distributed to all trainers
    '''

    def __init__(self, nclasses, nranks, rank_id, layer_helper):
        self.nclasses = nclasses
        self.nranks = nranks
        self.rank_id = rank_id
        self._layer_helper = layer_helper

        self.shard_dim = (nclasses + nranks - 1) // nranks
        self.padding_dim = 0
        self.is_equal_division = True
        if nclasses % nranks != 0:
            self.is_equal_division = False
            if rank_id == nranks - 1:
                other_shard_dim = self.shard_dim
                self.shard_dim = nclasses % other_shard_dim
                self.padding_dim = other_shard_dim - self.shard_dim

    def create_parameter(self,
                         dtype,
                         in_dim,
                         param_attr=None,
                         transpose_weight=False,
                         use_bias=True):
        if param_attr is None:
            stdv = math.sqrt(2.0 / (in_dim + self.nclasses))
            param_attr = ParamAttr(initializer=Normal(scale=stdv))
        weight_shape = [self.shard_dim, in_dim
                        ] if transpose_weight else [in_dim, self.shard_dim]
        weight = self._layer_helper.create_parameter(
            shape=weight_shape, dtype=dtype, attr=param_attr, is_bias=False)
        # avoid distributed parameter allreduce gradients
        weight.is_distributed = True
        # avoid distributed parameter broadcasting in startup program
        default_startup_program().global_block().vars[
            weight.name].is_distributed = True

        bias = None
        if use_bias:
            bias = self._layer_helper.create_parameter(
                shape=[self.shard_dim],
                attr=ParamAttr(),
                dtype=dtype,
                is_bias=True)
            bias.is_distributed = True
            default_startup_program().global_block().vars[
                bias.name].is_distributed = True
        return weight, bias

    def softmax_with_cross_entropy(self, shard_logit, shard_label):
        shard_max = nn.reduce_max(shard_logit, dim=1, keep_dim=True)
        global_max = _c_allreduce(
            shard_max, reduce_type='max', use_calc_stream=True)
        shard_logit_new = nn.elementwise_sub(shard_logit, global_max)

        shard_exp = ops.exp(shard_logit_new)
        shard_demon = nn.reduce_sum(shard_exp, dim=1, keep_dim=True)
        global_demon = _c_allreduce(
            shard_demon, reduce_type='sum', use_calc_stream=True)

        global_log_demon = nn.log(global_demon)
        shard_log_prob = shard_logit_new - global_log_demon
        shard_prob = ops.exp(shard_log_prob)

        shard_one_hot = nn.one_hot(
            shard_label, depth=self.shard_dim, allow_out_of_range=True)
        target_log_prob = nn.reduce_min(
            shard_log_prob * shard_one_hot, dim=1, keep_dim=True)
        shard_loss = nn.scale(target_log_prob, scale=-1.0)
        global_loss = _c_reducescatter(
            shard_loss, nranks=self.nranks, use_calc_stream=True)
        return global_loss, shard_prob

    def fc_classify(self, x, label, param_attr=None, use_bias=True):
        flatten_dim = reduce(lambda a, b: a * b, x.shape[1:], 1)
        weight, bias = self.create_parameter(
            dtype=x.dtype,
            in_dim=flatten_dim,
            param_attr=param_attr,
            use_bias=use_bias)

        x_all = _c_allgather(x, nranks=self.nranks, use_calc_stream=True)
        label_all = _c_allgather(
            label, nranks=self.nranks, use_calc_stream=True)
        label_all.stop_gradient = True

        shard_fc = nn.mul(x_all, weight, x_num_col_dims=1)
        if use_bias:
            shard_fc = nn.elementwise_add(shard_fc, bias)

        shard_label = nn.shard_index(
            label_all,
            index_num=self.nclasses,
            nshards=self.nranks,
            shard_id=self.rank_id,
            ignore_value=-1)
        shard_label.stop_gradient = True

        global_loss, shard_prob = self.softmax_with_cross_entropy(shard_fc,
                                                                  shard_label)
        avg_loss = nn.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_fc)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('shard_label', shard_label)
        avg_loss._set_info('shard_dim', self.shard_dim)

        return avg_loss

    def arcmargin_classify(self,
                           x,
                           label,
                           margin=0.5,
                           logit_scale=64,
                           param_attr=None):
        '''
        reference: ArcFace. https://arxiv.org/abs/1801.07698
        '''
        flatten_dim = reduce(lambda a, b: a * b, x.shape[1:], 1)
        weight, bias = self.create_parameter(
            dtype=x.dtype,
            in_dim=flatten_dim,
            param_attr=param_attr,
            transpose_weight=True,
            use_bias=False)

        # normalize x
        x_l2 = ops.sqrt(nn.reduce_sum(nn.square(x), dim=1))
        norm_x = nn.elementwise_div(x, x_l2, axis=0)

        norm_x_all = _c_allgather(
            norm_x, nranks=self.nranks, use_calc_stream=True)
        label_all = _c_allgather(
            label, nranks=self.nranks, use_calc_stream=True)
        label_all.stop_gradient = True
        shard_label = nn.shard_index(
            label_all,
            index_num=self.nclasses,
            nshards=self.nranks,
            shard_id=self.rank_id,
            ignore_value=-1)
        shard_label.stop_gradient = True

        # normalize weight
        weight_l2 = ops.sqrt(nn.reduce_sum(nn.square(weight), dim=1))
        norm_weight = nn.elementwise_div(weight, weight_l2, axis=0)
        norm_weight = nn.transpose(norm_weight, perm=[1, 0])

        shard_cos = nn.mul(norm_x_all, norm_weight, x_num_col_dims=1)

        theta = ops.acos(shard_cos)
        margin_cos = ops.cos(theta + margin)

        shard_one_hot = nn.one_hot(
            shard_label, depth=self.shard_dim, allow_out_of_range=True)
        shard_one_hot.stop_gradient = True

        diff = (margin_cos - shard_cos) * shard_one_hot
        shard_target_cos = shard_cos + diff
        shard_logit = nn.scale(shard_target_cos, scale=logit_scale)

        global_loss, shard_prob = self.softmax_with_cross_entropy(shard_logit,
                                                                  shard_label)
        avg_loss = nn.mean(global_loss)

        avg_loss._set_info('shard_logit', shard_logit)
        avg_loss._set_info('shard_prob', shard_prob)
        avg_loss._set_info('shard_label', shard_label)
        avg_loss._set_info('shard_dim', self.shard_dim)

        return avg_loss


def distributed_fc_classify(x,
                            label,
                            class_num,
                            nranks,
                            rank_id,
                            param_attr=None,
                            use_bias=True,
                            name='dist_fc'):
    '''
    '''
    helper = LayerHelper(name, **locals())
    classifier = DistributedClassifier(class_num, nranks, rank_id, helper)
    return classifier.fc_classify(x, label, param_attr, use_bias)


def distributed_arcmargin_classify(x,
                                   label,
                                   class_num,
                                   nranks,
                                   rank_id,
                                   margin=0.5,
                                   logit_scale=64,
                                   param_attr=None,
                                   name='dist_fc'):
    '''
    '''
    helper = LayerHelper(name, **locals())
    classifier = DistributedClassifier(class_num, nranks, rank_id, helper)
    return classifier.arcmargin_classify(
        x=x,
        label=label,
        margin=margin,
        logit_scale=logit_scale,
        param_attr=param_attr)


def distributed_fc(x,
                   out_dim,
                   nranks,
                   rank_id,
                   param_attr=None,
                   use_bias=True,
                   name='dist_fc'):
    '''
    '''
    helper = LayerHelper(name, **locals())
    classifier = DistributedClassifier(out_dim, nranks, rank_id, helper)
    weight, bias = classifier.create_parameter(
        dtype=x.dtype,
        in_dim=x.shape[-1],
        param_attr=param_attr,
        use_bias=use_bias)
    x_all = _c_allgather(x, nranks=self.nranks, use_calc_stream=True)
    label_all = _c_allgather(label, nranks=self.nranks, use_calc_stream=True)

    shard_fc = nn.mul(x_all, weight)
    if use_bias:
        shard_fc = nn.elementwise_add(shard_fc, bias)

    # sample code
    #if not classifier.is_equal_division:
    #    shard_fc = nn.pad(shard_fc)
    #fc = _c_slice_allgather(shard_fc,
    #                        nranks=nranks,
    #                        rank_id=rank_id)
    #if not classifier.is_equal_division:
    #    fc = nn.depad(fc)
    #return fc
    raise NotImplementedError('distributed_fc')
