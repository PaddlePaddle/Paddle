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
import unittest

import paddle.fluid.layers as layers
import paddle.fluid.nets as nets
from paddle.fluid.framework import Program, program_guard, default_main_program
from paddle.fluid.param_attr import ParamAttr
import decorators


class TestBook(unittest.TestCase):
    def test_fit_a_line(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_recognize_digits_mlp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            # Change g_program, so the rest layers use `g_program`
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(input=[hidden2, hidden1],
                                size=10,
                                act='softmax',
                                param_attr=["sftmax.w1", "sftmax.w2"])
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_simple_conv2d(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(name='pixel', shape=[3, 48, 48], dtype='int32')
            layers.conv2d(input=images, num_filters=3, filter_size=[4, 4])

        print(str(program))

    def test_conv2d_transpose(self):
        program = Program()
        with program_guard(program):
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            layers.conv2d_transpose(input=img, num_filters=10, output_size=28)
        print(str(program))

    def test_recognize_digits_conv(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu")
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu")

            predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)

        print(str(program))

    def test_word_embedding(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            dict_size = 10000
            embed_size = 32
            first_word = layers.data(name='firstw', shape=[1], dtype='int64')
            second_word = layers.data(name='secondw', shape=[1], dtype='int64')
            third_word = layers.data(name='thirdw', shape=[1], dtype='int64')
            forth_word = layers.data(name='forthw', shape=[1], dtype='int64')
            next_word = layers.data(name='nextw', shape=[1], dtype='int64')

            embed_first = layers.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_second = layers.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            embed_third = layers.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_forth = layers.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            concat_embed = layers.concat(
                input=[embed_first, embed_second, embed_third, embed_forth],
                axis=1)

            hidden1 = layers.fc(input=concat_embed, size=256, act='sigmoid')
            predict_word = layers.fc(input=hidden1,
                                     size=dict_size,
                                     act='softmax')
            cost = layers.cross_entropy(input=predict_word, label=next_word)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_linear_chain_crf(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            label_dict_len = 10
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden = layers.fc(input=images, size=128)
            crf = layers.linear_chain_crf(
                input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
            crf_decode = layers.crf_decoding(
                input=hidden, param_attr=ParamAttr(name="crfw"))
            layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) / 2)
            self.assertFalse(crf is None)
            self.assertFalse(crf_decode is None)

        print(str(program))

    def test_sigmoid_cross_entropy(self):
        program = Program()
        with program_guard(program):
            dat = layers.data(name='data', shape=[10], dtype='float32')
            lbl = layers.data(name='label', shape=[10], dtype='float32')
            self.assertIsNotNone(
                layers.sigmoid_cross_entropy_with_logits(
                    x=dat, label=lbl))
        print(str(program))

    def test_sequence_expand(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            self.assertIsNotNone(layers.sequence_expand(x=x, y=y, ref_level=1))
        print(str(program))

    def test_lstm_unit(self):
        program = Program()
        with program_guard(program):
            x_t_data = layers.data(
                name='x_t_data', shape=[10, 10], dtype='float32')
            x_t = layers.fc(input=x_t_data, size=10)
            prev_hidden_data = layers.data(
                name='prev_hidden_data', shape=[10, 30], dtype='float32')
            prev_hidden = layers.fc(input=prev_hidden_data, size=30)
            prev_cell_data = layers.data(
                name='prev_cell', shape=[10, 30], dtype='float32')
            prev_cell = layers.fc(input=prev_cell_data, size=30)
            self.assertIsNotNone(
                layers.lstm_unit(
                    x_t=x_t, hidden_t_prev=prev_hidden, cell_t_prev=prev_cell))
        print(str(program))

    def test_dynamic_lstmp(self):
        program = Program()
        with program_guard(program):
            hidden_dim, proj_dim = 16, 8
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            fc_out = layers.fc(input=seq_data, size=4 * hidden_dim)
            self.assertIsNotNone(
                layers.dynamic_lstmp(
                    input=fc_out, size=4 * hidden_dim, proj_size=proj_dim))
        print(str(program))

    def test_sequence_softmax(self):
        program = Program()
        with program_guard(program):
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            seq = layers.fc(input=seq_data, size=20)
            self.assertIsNotNone(layers.sequence_softmax(seq))
        print(str(program))

    def test_softmax(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[10], dtype='float32')
            hid = layers.fc(input=data, size=20)
            self.assertIsNotNone(layers.softmax(hid))
        print(str(program))

    def test_lrn(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[6, 2, 2], dtype='float32')
            self.assertIsNotNone(layers.lrn(data))
        print(str(program))

    def test_get_places(self):
        program = Program()
        with program_guard(program):
            x = layers.get_places(device_count=4)
            self.assertIsNotNone(x)
        print(str(program))

    def test_sequence_reshape(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8], dtype='float32', lod_level=1)
            out = layers.sequence_reshape(input=x, new_dim=16)
            self.assertIsNotNone(out)
        print(str(program))

    def test_im2sequence(self):
        print("test_im2sequence")
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 128, 128], dtype='float32')
            output = layers.im2sequence(
                input=x, stride=[1, 1], filter_size=[2, 2])
            self.assertIsNotNone(output)
        print(str(program))

    @decorators.prog_scope()
    def test_nce(self):
        window_size = 5
        words = []
        for i in xrange(window_size):
            words.append(
                layers.data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

        dict_size = 10000
        label_word = int(window_size / 2) + 1

        embs = []
        for i in xrange(window_size):
            if i == label_word:
                continue

            emb = layers.embedding(
                input=words[i],
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=True)

            embs.append(emb)

        embs = layers.concat(input=embs, axis=1)
        loss = layers.nce(input=embs,
                          label=words[label_word],
                          num_total_classes=dict_size,
                          param_attr='nce.w',
                          bias_attr='nce.b')
        avg_loss = layers.mean(loss)
        self.assertIsNotNone(avg_loss)
        print(str(default_main_program()))

    def test_row_conv(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32', lod_level=1)
            out = layers.row_conv(input=x, future_context_size=2)
            self.assertIsNotNone(out)
        print(str(program))

    def test_multiplex(self):
        program = Program()
        with program_guard(program):
            x1 = layers.data(name='x1', shape=[4], dtype='float32')
            x2 = layers.data(name='x2', shape=[4], dtype='float32')
            index = layers.data(name='index', shape=[1], dtype='int32')
            out = layers.multiplex(inputs=[x1, x2], index=index)
            self.assertIsNotNone(out)
        print(str(program))

    def test_softmax_with_cross_entropy(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32')
            y = layers.data(name='label', shape=[1], dtype='int64')
            loss = layers.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)
        print(str(program))

    def test_smooth_l1(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='label', shape=[4], dtype='float32')
            loss = layers.smooth_l1(x, y)
            self.assertIsNotNone(loss)
        print(str(program))

    def test_lod_reset(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            print(layers.lod_reset(x=x, y=y))
        print(str(program))

    def test_label_smooth(self):
        program = Program()
        with program_guard(program):
            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
            self.assertIsNotNone(smooth_label)
        print(str(program))

    def test_topk(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name="label", shape=[200], dtype="float32")
            values, indices = layers.topk(data, k=5)
            self.assertIsNotNone(values)
            self.assertIsNotNone(indices)
        print(str(program))

    def test_roi_pool(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[256, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            output = layers.roi_pool(x, rois, 7, 7, 0.6)
            self.assertIsNotNone(output)
        print(str(program))

    def test_resize_bilinear(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_bilinear(x, out_shape=[12, 12])
            self.assertIsNotNone(output)
            output = layers.resize_bilinear(x, scale=3)
            self.assertIsNotNone(output)
        print(str(program))

    def test_polygon_box_transform(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8, 4, 4], dtype="float32")
            output = layers.polygon_box_transform(input=x)
            self.assertIsNotNone(output)
        print(str(program))

    def test_l2_normalize(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8, 7, 10], dtype="float32")
            output = layers.l2_normalize(x, axis=1)

    def test_maxout(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='x', shape=[8, 6, 6], dtype="float32")
            output = layers.maxout(x=data, groups=2)
            self.assertIsNotNone(output)
        print(str(program))

    def test_crop(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 5], dtype="float32")
            y = layers.data(name='y', shape=[2, 3], dtype="float32")
            output = layers.crop(x, shape=y)
            self.assertIsNotNone(output)
        print(str(program))

    def test_mean_iou(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32')
            y = layers.data(name='label', shape=[1], dtype='int64')
            iou = layers.mean_iou(x, y, 2)
            self.assertIsNotNone(iou)
        print(str(program))


if __name__ == '__main__':
    unittest.main()
