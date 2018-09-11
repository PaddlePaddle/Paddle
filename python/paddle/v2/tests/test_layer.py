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

import unittest

import paddle.v2.activation as activation
import paddle.v2.attr as attr
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
import paddle.v2.pooling as pooling
import paddle.v2.networks as networks
import paddle.v2.evaluator as evaluator

pixel = layer.data(name='pixel', type=data_type.dense_vector(128))
label = layer.data(name='label', type=data_type.integer_value(10))
weight = layer.data(name='weight', type=data_type.dense_vector(1))
combine_weight = layer.data(
    name='weight_combine', type=data_type.dense_vector(10))
score = layer.data(name='score', type=data_type.dense_vector(1))

hidden = layer.fc(input=pixel,
                  size=100,
                  act=activation.Sigmoid(),
                  param_attr=attr.Param(name='hidden'))
inference = layer.fc(input=hidden, size=10, act=activation.Softmax())
conv = layer.img_conv(
    input=pixel,
    filter_size=1,
    filter_size_y=1,
    num_channels=8,
    num_filters=16,
    act=activation.Linear())


class ImageLayerTest(unittest.TestCase):
    def test_conv_layer(self):
        conv_shift = layer.conv_shift(a=pixel, b=score)
        print layer.parse_network(conv, conv_shift)

    def test_pooling_layer(self):
        maxpool = layer.img_pool(
            input=conv,
            pool_size=2,
            num_channels=16,
            padding=1,
            pool_type=pooling.Max())
        spp = layer.spp(input=conv,
                        pyramid_height=2,
                        num_channels=16,
                        pool_type=pooling.Max())
        maxout = layer.maxout(input=conv, num_channels=16, groups=4)
        print layer.parse_network([maxpool, spp, maxout])

    def test_norm_layer(self):
        norm1 = layer.img_cmrnorm(input=conv, size=5)
        norm2 = layer.batch_norm(input=conv)
        norm3 = layer.sum_to_one_norm(input=conv)
        print layer.parse_network([norm1, norm2, norm3])


class AggregateLayerTest(unittest.TestCase):
    def test_aggregate_layer(self):
        pool = layer.pooling(
            input=pixel,
            pooling_type=pooling.Avg(),
            agg_level=layer.AggregateLevel.TO_SEQUENCE)
        last_seq = layer.last_seq(input=pixel)
        first_seq = layer.first_seq(input=pixel)
        concat = layer.concat(input=[last_seq, first_seq])
        seq_concat = layer.seq_concat(a=last_seq, b=first_seq)
        print layer.parse_network(
            [pool, last_seq, first_seq, concat, seq_concat])


class MathLayerTest(unittest.TestCase):
    def test_math_layer(self):
        addto = layer.addto(input=[pixel, pixel])
        linear_comb = layer.linear_comb(
            weights=combine_weight, vectors=hidden, size=10)
        interpolation = layer.interpolation(
            input=[hidden, hidden], weight=score)
        bilinear = layer.bilinear_interp(input=conv, out_size_x=4, out_size_y=4)
        power = layer.power(input=pixel, weight=score)
        scaling = layer.scaling(input=pixel, weight=score)
        slope = layer.slope_intercept(input=pixel)
        tensor = layer.tensor(a=pixel, b=pixel, size=1000)
        cos_sim = layer.cos_sim(a=pixel, b=pixel)
        trans = layer.trans(input=tensor)
        print layer.parse_network([
            addto, linear_comb, interpolation, power, scaling, slope, tensor,
            cos_sim, trans
        ])


class ReshapeLayerTest(unittest.TestCase):
    def test_reshape_layer(self):
        block_expand = layer.block_expand(
            input=conv, num_channels=4, stride_x=1, block_x=1)
        expand = layer.expand(
            input=weight,
            expand_as=pixel,
            expand_level=layer.ExpandLevel.FROM_NO_SEQUENCE)
        repeat = layer.repeat(input=pixel, num_repeats=4)
        reshape = layer.seq_reshape(input=pixel, reshape_size=4)
        rotate = layer.rotate(input=pixel, height=16, width=49)
        print layer.parse_network(
            [block_expand, expand, repeat, reshape, rotate])


class RecurrentLayerTest(unittest.TestCase):
    def test_recurrent_layer(self):
        word = layer.data(name='word', type=data_type.integer_value(12))
        recurrent = layer.recurrent(input=word)
        lstm = layer.lstmemory(input=word)
        gru = layer.grumemory(input=word)
        print layer.parse_network([recurrent, lstm, gru])


class CostLayerTest(unittest.TestCase):
    def test_cost_layer(self):
        cost1 = layer.classification_cost(input=inference, label=label)
        cost2 = layer.classification_cost(
            input=inference, label=label, weight=weight)
        cost3 = layer.cross_entropy_cost(input=inference, label=label)
        cost4 = layer.cross_entropy_with_selfnorm_cost(
            input=inference, label=label)
        cost5 = layer.square_error_cost(input=inference, label=label)
        cost6 = layer.square_error_cost(
            input=inference, label=label, weight=weight)
        cost7 = layer.multi_binary_label_cross_entropy_cost(
            input=inference, label=label)
        cost8 = layer.rank_cost(left=score, right=score, label=score)
        cost9 = layer.lambda_cost(input=inference, score=score)
        cost10 = layer.sum_cost(input=inference)
        cost11 = layer.huber_regression_cost(input=score, label=label)
        cost12 = layer.huber_classification_cost(input=score, label=label)

        print layer.parse_network([cost1, cost2])
        print layer.parse_network([cost3, cost4])
        print layer.parse_network([cost5, cost6])
        print layer.parse_network([cost7, cost8, cost9, cost10, cost11, cost12])

        crf = layer.crf(input=inference, label=label)
        crf_decoding = layer.crf_decoding(input=inference, size=3)
        ctc = layer.ctc(input=inference, label=label)
        warp_ctc = layer.warp_ctc(input=pixel, label=label)
        nce = layer.nce(input=inference, label=label, num_classes=3)
        hsigmoid = layer.hsigmoid(input=inference, label=label, num_classes=3)

        print layer.parse_network(
            [crf, crf_decoding, ctc, warp_ctc, nce, hsigmoid])


class OtherLayerTest(unittest.TestCase):
    def test_sampling_layer(self):
        maxid = layer.max_id(input=inference)
        sampling_id = layer.sampling_id(input=inference)
        eos = layer.eos(input=maxid, eos_id=5)
        layer.printer(maxid)
        print layer.parse_network([maxid, sampling_id, eos])

    def test_slicing_joining_layer(self):
        pad = layer.pad(input=conv, pad_c=[2, 3], pad_h=[1, 2], pad_w=[3, 1])
        print layer.parse_network(pad)


class ProjOpTest(unittest.TestCase):
    def test_projection(self):
        input = layer.data(name='data2', type=data_type.dense_vector(784))
        word = layer.data(
            name='word2', type=data_type.integer_value_sequence(10000))
        fc0 = layer.fc(input=input, size=100, act=activation.Sigmoid())
        fc1 = layer.fc(input=input, size=200, act=activation.Sigmoid())
        mixed0 = layer.mixed(
            size=256,
            input=[
                layer.full_matrix_projection(input=fc0),
                layer.full_matrix_projection(input=fc1)
            ])
        with layer.mixed(size=200) as mixed1:
            mixed1 += layer.full_matrix_projection(input=fc0)
            mixed1 += layer.identity_projection(input=fc1)

        table = layer.table_projection(input=word)
        emb0 = layer.mixed(size=512, input=table)
        with layer.mixed(size=512) as emb1:
            emb1 += table

        scale = layer.scaling_projection(input=fc0)
        scale0 = layer.mixed(size=100, input=scale)
        with layer.mixed(size=100) as scale1:
            scale1 += scale

        dotmul = layer.dotmul_projection(input=fc0)
        dotmul0 = layer.mixed(size=100, input=dotmul)
        with layer.mixed(size=100) as dotmul1:
            dotmul1 += dotmul

        context = layer.context_projection(input=fc0, context_len=5)
        context0 = layer.mixed(size=500, input=context)
        with layer.mixed(size=500) as context1:
            context1 += context

        conv = layer.conv_projection(
            input=input,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = layer.mixed(input=conv, bias_attr=True)
        with layer.mixed(bias_attr=True) as conv1:
            conv1 += conv

        print layer.parse_network(mixed0)
        print layer.parse_network(mixed1)
        print layer.parse_network(emb0)
        print layer.parse_network(emb1)
        print layer.parse_network(scale0)
        print layer.parse_network(scale1)
        print layer.parse_network(dotmul0)
        print layer.parse_network(dotmul1)
        print layer.parse_network(conv0)
        print layer.parse_network(conv1)

    def test_operator(self):
        ipt0 = layer.data(name='data1', type=data_type.dense_vector(784))
        ipt1 = layer.data(name='word1', type=data_type.dense_vector(128))
        fc0 = layer.fc(input=ipt0, size=100, act=activation.Sigmoid())
        fc1 = layer.fc(input=ipt0, size=100, act=activation.Sigmoid())

        dotmul_op = layer.dotmul_operator(a=fc0, b=fc1)
        dotmul0 = layer.mixed(input=dotmul_op)
        with layer.mixed() as dotmul1:
            dotmul1 += dotmul_op

        conv = layer.conv_operator(
            img=ipt0,
            filter=ipt1,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = layer.mixed(input=conv)
        with layer.mixed() as conv1:
            conv1 += conv

        print layer.parse_network(dotmul0)
        print layer.parse_network(dotmul1)
        print layer.parse_network(conv0)
        print layer.parse_network(conv1)


class NetworkTests(unittest.TestCase):
    def test_vgg(self):
        img = layer.data(name='pixel1', type=data_type.dense_vector(784))
        vgg_out = networks.small_vgg(
            input_image=img, num_channels=1, num_classes=2)
        print layer.parse_network(vgg_out)


class EvaluatorTest(unittest.TestCase):
    def test_evaluator(self):
        img = layer.data(name='pixel2', type=data_type.dense_vector(784))
        output = layer.fc(input=img,
                          size=10,
                          act=activation.Softmax(),
                          name='fc_here')
        lbl = layer.data(name='label2', type=data_type.integer_value(10))
        cost = layer.cross_entropy_cost(input=output, label=lbl)

        evaluator.classification_error(input=output, label=lbl)
        print layer.parse_network(cost)
        print layer.parse_network(output)


if __name__ == '__main__':
    unittest.main()
