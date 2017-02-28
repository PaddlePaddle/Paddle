# Copyright PaddlePaddle contributors. All Rights Reserved
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

import paddle.v2.configs as configs

pixel = configs.layer.data(
    name='pixel', type=configs.data_type.dense_vector(128))
label = configs.layer.data(
    name='label', type=configs.data_type.integer_value(10))
weight = configs.layer.data(
    name='weight', type=configs.data_type.dense_vector(10))
score = configs.layer.data(name='score', type=configs.data_type.dense_vector(1))

hidden = configs.layer.fc(input=pixel,
                          size=100,
                          act=configs.activation.Sigmoid(),
                          param_attr=configs.attr.ParamAttr(name='hidden'))
inference = configs.layer.fc(input=hidden,
                             size=10,
                             act=configs.activation.Softmax())
conv = configs.layer.img_conv(
    input=pixel,
    filter_size=1,
    filter_size_y=1,
    num_channels=8,
    num_filters=16,
    act=configs.activation.Linear())


class ImageLayerTest(unittest.TestCase):
    def test_conv_layer(self):
        conv_shift = configs.layer.conv_shift(a=pixel, b=score)
        print configs.layer.parse_network(conv, conv_shift)

    def test_pooling_layer(self):
        maxpool = configs.layer.img_pool(
            input=conv,
            pool_size=2,
            num_channels=16,
            padding=1,
            pool_type=configs.pooling.Max())
        spp = configs.layer.spp(input=conv,
                                pyramid_height=2,
                                num_channels=16,
                                pool_type=configs.pooling.Max())
        maxout = configs.layer.maxout(input=conv, num_channels=16, groups=4)
        print configs.layer.parse_network(maxpool, spp, maxout)

    def test_norm_layer(self):
        norm1 = configs.layer.img_cmrnorm(input=conv, size=5)
        norm2 = configs.layer.batch_norm(input=conv)
        norm3 = configs.layer.sum_to_one_norm(input=conv)
        print configs.layer.parse_network(norm1, norm2, norm3)


class AggregateLayerTest(unittest.TestCase):
    def test_aggregate_layer(self):
        pool = configs.layer.pooling(
            input=pixel,
            pooling_type=configs.pooling.Avg(),
            agg_level=configs.layer.AggregateLevel.EACH_SEQUENCE)
        last_seq = configs.layer.last_seq(input=pixel)
        first_seq = configs.layer.first_seq(input=pixel)
        concat = configs.layer.concat(input=[last_seq, first_seq])
        seq_concat = configs.layer.seq_concat(a=last_seq, b=first_seq)
        print configs.layer.parse_network(pool, last_seq, first_seq, concat,
                                          seq_concat)


class MathLayerTest(unittest.TestCase):
    def test_math_layer(self):
        addto = configs.layer.addto(input=[pixel, pixel])
        linear_comb = configs.layer.linear_comb(
            weights=weight, vectors=hidden, size=10)
        interpolation = configs.layer.interpolation(
            input=[hidden, hidden], weight=score)
        bilinear = configs.layer.bilinear_interp(
            input=conv, out_size_x=4, out_size_y=4)
        power = configs.layer.power(input=pixel, weight=score)
        scaling = configs.layer.scaling(input=pixel, weight=score)
        slope = configs.layer.slope_intercept(input=pixel)
        tensor = configs.layer.tensor(a=pixel, b=pixel, size=1000)
        cos_sim = configs.layer.cos_sim(a=pixel, b=pixel)
        trans = configs.layer.trans(input=tensor)
        print configs.layer.parse_network(addto, linear_comb, interpolation,
                                          power, scaling, slope, tensor,
                                          cos_sim, trans)


class ReshapeLayerTest(unittest.TestCase):
    def test_reshape_layer(self):
        block_expand = configs.layer.block_expand(
            input=conv, num_channels=4, stride_x=1, block_x=1)
        expand = configs.layer.expand(
            input=weight,
            expand_as=pixel,
            expand_level=configs.layer.ExpandLevel.FROM_TIMESTEP)
        repeat = configs.layer.repeat(input=pixel, num_repeats=4)
        reshape = configs.layer.seq_reshape(input=pixel, reshape_size=4)
        rotate = configs.layer.rotate(input=pixel, height=16, width=49)
        print configs.layer.parse_network(block_expand, expand, repeat, reshape,
                                          rotate)


class RecurrentLayerTest(unittest.TestCase):
    def test_recurrent_layer(self):
        word = configs.layer.data(
            name='word', type=configs.data_type.integer_value(12))
        recurrent = configs.layer.recurrent(input=word)
        lstm = configs.layer.lstmemory(input=word)
        gru = configs.layer.grumemory(input=word)
        print configs.layer.parse_network(recurrent, lstm, gru)


class CostLayerTest(unittest.TestCase):
    def test_cost_layer(self):
        cost1 = configs.layer.classification_cost(input=inference, label=label)
        cost2 = configs.layer.classification_cost(
            input=inference, label=label, weight=weight)
        cost3 = configs.layer.cross_entropy_cost(input=inference, label=label)
        cost4 = configs.layer.cross_entropy_with_selfnorm_cost(
            input=inference, label=label)
        cost5 = configs.layer.regression_cost(input=inference, label=label)
        cost6 = configs.layer.regression_cost(
            input=inference, label=label, weight=weight)
        cost7 = configs.layer.multi_binary_label_cross_entropy_cost(
            input=inference, label=label)
        cost8 = configs.layer.rank_cost(left=score, right=score, label=score)
        cost9 = configs.layer.lambda_cost(input=inference, score=score)
        cost10 = configs.layer.sum_cost(input=inference)
        cost11 = configs.layer.huber_cost(input=score, label=label)

        print configs.layer.parse_network(cost1, cost2)
        print configs.layer.parse_network(cost3, cost4)
        print configs.layer.parse_network(cost5, cost6)
        print configs.layer.parse_network(cost7, cost8, cost9, cost10, cost11)

        crf = configs.layer.crf(input=inference, label=label)
        crf_decoding = configs.layer.crf_decoding(input=inference, size=3)
        ctc = configs.layer.ctc(input=inference, label=label)
        warp_ctc = configs.layer.warp_ctc(input=pixel, label=label)
        nce = configs.layer.nce(input=inference, label=label, num_classes=3)
        hsigmoid = configs.layer.hsigmoid(
            input=inference, label=label, num_classes=3)

        print configs.layer.parse_network(crf, crf_decoding, ctc, warp_ctc, nce,
                                          hsigmoid)


class OtherLayerTest(unittest.TestCase):
    def test_sampling_layer(self):
        maxid = configs.layer.max_id(input=inference)
        sampling_id = configs.layer.sampling_id(input=inference)
        eos = configs.layer.eos(input=maxid, eos_id=5)
        print configs.layer.parse_network(maxid, sampling_id, eos)

    def test_slicing_joining_layer(self):
        pad = configs.layer.pad(input=conv,
                                pad_c=[2, 3],
                                pad_h=[1, 2],
                                pad_w=[3, 1])
        print configs.layer.parse_network(pad)


class ProjOpTest(unittest.TestCase):
    def test_projection(self):
        input = configs.layer.data(
            name='data', type=configs.data_type.dense_vector(784))
        word = configs.layer.data(
            name='word', type=configs.data_type.integer_value_sequence(10000))
        fc0 = configs.layer.fc(input=input,
                               size=100,
                               act=configs.activation.Sigmoid())
        fc1 = configs.layer.fc(input=input,
                               size=200,
                               act=configs.activation.Sigmoid())
        mixed0 = configs.layer.mixed(
            size=256,
            input=[
                configs.layer.full_matrix_projection(input=fc0),
                configs.layer.full_matrix_projection(input=fc1)
            ])
        with configs.layer.mixed(size=200) as mixed1:
            mixed1 += configs.layer.full_matrix_projection(input=fc0)
            mixed1 += configs.layer.identity_projection(input=fc1)

        table = configs.layer.table_projection(input=word)
        emb0 = configs.layer.mixed(size=512, input=table)
        with configs.layer.mixed(size=512) as emb1:
            emb1 += table

        scale = configs.layer.scaling_projection(input=fc0)
        scale0 = configs.layer.mixed(size=100, input=scale)
        with configs.layer.mixed(size=100) as scale1:
            scale1 += scale

        dotmul = configs.layer.dotmul_projection(input=fc0)
        dotmul0 = configs.layer.mixed(size=100, input=dotmul)
        with configs.layer.mixed(size=100) as dotmul1:
            dotmul1 += dotmul

        context = configs.layer.context_projection(input=fc0, context_len=5)
        context0 = configs.layer.mixed(size=100, input=context)
        with configs.layer.mixed(size=100) as context1:
            context1 += context

        conv = configs.layer.conv_projection(
            input=input,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = configs.layer.mixed(input=conv, bias_attr=True)
        with configs.layer.mixed(bias_attr=True) as conv1:
            conv1 += conv

        print configs.layer.parse_network(mixed0)
        print configs.layer.parse_network(mixed1)
        print configs.layer.parse_network(emb0)
        print configs.layer.parse_network(emb1)
        print configs.layer.parse_network(scale0)
        print configs.layer.parse_network(scale1)
        print configs.layer.parse_network(dotmul0)
        print configs.layer.parse_network(dotmul1)
        print configs.layer.parse_network(conv0)
        print configs.layer.parse_network(conv1)

    def test_operator(self):
        ipt0 = configs.layer.data(
            name='data', type=configs.data_type.dense_vector(784))
        ipt1 = configs.layer.data(
            name='word', type=configs.data_type.dense_vector(128))
        fc0 = configs.layer.fc(input=ipt0,
                               size=100,
                               act=configs.activation.Sigmoid())
        fc1 = configs.layer.fc(input=ipt0,
                               size=100,
                               act=configs.activation.Sigmoid())

        dotmul_op = configs.layer.dotmul_operator(a=fc0, b=fc1)
        dotmul0 = configs.layer.mixed(input=dotmul_op)
        with configs.layer.mixed() as dotmul1:
            dotmul1 += dotmul_op

        conv = configs.layer.conv_operator(
            img=ipt0,
            filter=ipt1,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = configs.layer.mixed(input=conv)
        with configs.layer.mixed() as conv1:
            conv1 += conv

        print configs.layer.parse_network(dotmul0)
        print configs.layer.parse_network(dotmul1)
        print configs.layer.parse_network(conv0)
        print configs.layer.parse_network(conv1)


if __name__ == '__main__':
    unittest.main()
