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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers import detection
from paddle.fluid.framework import Program, program_guard
import unittest
import contextlib
from unittests.test_imperative_base import new_program_scope
from paddle.fluid import core
import paddle

paddle.enable_static()


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        with new_program_scope():
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(
            fluid.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield


class TestDetection(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            loc = layers.data(
                name='target_box',
                shape=[2, 10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            scores = layers.data(
                name='scores',
                shape=[2, 10, 20],
                append_batch_size=False,
                dtype='float32',
            )
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv
            )
            out2, index = layers.detection_output(
                scores=scores,
                loc=loc,
                prior_box=pb,
                prior_box_var=pbv,
                return_index=True,
            )
            self.assertIsNotNone(out)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(index)
            self.assertEqual(out.shape[-1], 6)
        print(str(program))

    def test_box_coder_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y,
                code_type='encode_center_size',
            )
            self.assertIsNotNone(bcoder)
        print(str(program))

    def test_box_coder_error(self):
        program = Program()
        with program_guard(program):
            x1 = fluid.data(name='x1', shape=[10, 4], dtype='int32')
            y1 = fluid.data(
                name='y1', shape=[10, 4], dtype='float32', lod_level=1
            )
            x2 = fluid.data(name='x2', shape=[10, 4], dtype='float32')
            y2 = fluid.data(
                name='y2', shape=[10, 4], dtype='int32', lod_level=1
            )

            self.assertRaises(
                TypeError,
                layers.box_coder,
                prior_box=x1,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y1,
                code_type='encode_center_size',
            )
            self.assertRaises(
                TypeError,
                layers.box_coder,
                prior_box=x2,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y2,
                code_type='encode_center_size',
            )

    def test_detection_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='y', shape=[4], dtype='float32')
            z = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            iou = layers.iou_similarity(x=x, y=y)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=y,
                target_box=z,
                code_type='encode_center_size',
            )
            self.assertIsNotNone(iou)
            self.assertIsNotNone(bcoder)

            matched_indices, matched_dist = layers.bipartite_match(iou)
            self.assertIsNotNone(matched_indices)
            self.assertIsNotNone(matched_dist)

            gt = layers.data(
                name='gt', shape=[1, 1], dtype='int32', lod_level=1
            )
            trg, trg_weight = layers.target_assign(
                gt, matched_indices, mismatch_value=0
            )
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

            gt2 = layers.data(
                name='gt2', shape=[10, 4], dtype='float32', lod_level=1
            )
            trg, trg_weight = layers.target_assign(
                gt2, matched_indices, mismatch_value=0
            )
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

        print(str(program))

    def test_ssd_loss(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            loc = layers.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = layers.data(
                name='gt_box', shape=[4], lod_level=1, dtype='float32'
            )
            gt_label = layers.data(
                name='gt_label', shape=[1], lod_level=1, dtype='int32'
            )
            loss = layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
            self.assertIsNotNone(loss)
            self.assertEqual(loss.shape[-1], 1)
        print(str(program))


class TestPriorBox(unittest.TestCase):
    def test_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 224, 224]
            images = fluid.layers.data(
                name='pixel', shape=data_shape, dtype='float32'
            )
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.prior_box(
                input=conv1,
                image=images,
                min_sizes=[100.0],
                aspect_ratios=[1.0],
                flip=True,
                clip=True,
            )
            assert len(box.shape) == 4
            assert box.shape == var.shape
            assert box.shape[3] == 4


class TestPriorBox2(unittest.TestCase):
    def test_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [None, 3, None, None]
            images = fluid.data(name='pixel', shape=data_shape, dtype='float32')
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.prior_box(
                input=conv1,
                image=images,
                min_sizes=[100.0],
                aspect_ratios=[1.0],
                flip=True,
                clip=True,
            )
            assert len(box.shape) == 4
            assert box.shape == var.shape
            assert box.shape[3] == 4


class TestMultiBoxHead(unittest.TestCase):
    def test_multi_box_head(self):
        data_shape = [3, 224, 224]
        mbox_locs, mbox_confs, box, var = self.multi_box_head_output(data_shape)

        assert len(box.shape) == 2
        assert box.shape == var.shape
        assert box.shape[1] == 4
        assert mbox_locs.shape[1] == mbox_confs.shape[1]

    def multi_box_head_output(self, data_shape):
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32'
        )
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        conv2 = fluid.layers.conv2d(conv1, 3, 3, 2)
        conv3 = fluid.layers.conv2d(conv2, 3, 3, 2)
        conv4 = fluid.layers.conv2d(conv3, 3, 3, 2)
        conv5 = fluid.layers.conv2d(conv4, 3, 3, 2)

        mbox_locs, mbox_confs, box, var = layers.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
            image=images,
            num_classes=21,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[
                [2.0],
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0],
                [2.0],
            ],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True,
        )

        return mbox_locs, mbox_confs, box, var


class TestDetectionMAP(unittest.TestCase):
    def test_detection_map(self):
        program = Program()
        with program_guard(program):
            detect_res = layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32',
            )
            label = layers.data(
                name='label',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32',
            )

            map_out = detection.detection_map(detect_res, label, 21)
            self.assertIsNotNone(map_out)
            self.assertEqual(map_out.shape, (1,))
        print(str(program))


class TestRpnTargetAssign(unittest.TestCase):
    def test_rpn_target_assign(self):
        program = Program()
        with program_guard(program):
            bbox_pred_shape = [10, 50, 4]
            cls_logits_shape = [10, 50, 2]
            anchor_shape = [50, 4]

            bbox_pred = layers.data(
                name='bbox_pred',
                shape=bbox_pred_shape,
                append_batch_size=False,
                dtype='float32',
            )
            cls_logits = layers.data(
                name='cls_logits',
                shape=cls_logits_shape,
                append_batch_size=False,
                dtype='float32',
            )
            anchor_box = layers.data(
                name='anchor_box',
                shape=anchor_shape,
                append_batch_size=False,
                dtype='float32',
            )
            anchor_var = layers.data(
                name='anchor_var',
                shape=anchor_shape,
                append_batch_size=False,
                dtype='float32',
            )
            gt_boxes = layers.data(
                name='gt_boxes', shape=[4], lod_level=1, dtype='float32'
            )
            is_crowd = layers.data(
                name='is_crowd',
                shape=[1, 10],
                dtype='int32',
                lod_level=1,
                append_batch_size=False,
            )
            im_info = layers.data(
                name='im_info',
                shape=[1, 3],
                dtype='float32',
                lod_level=1,
                append_batch_size=False,
            )
            outs = layers.rpn_target_assign(
                bbox_pred=bbox_pred,
                cls_logits=cls_logits,
                anchor_box=anchor_box,
                anchor_var=anchor_var,
                gt_boxes=gt_boxes,
                is_crowd=is_crowd,
                im_info=im_info,
                rpn_batch_size_per_im=256,
                rpn_straddle_thresh=0.0,
                rpn_fg_fraction=0.5,
                rpn_positive_overlap=0.7,
                rpn_negative_overlap=0.3,
                use_random=False,
            )
            pred_scores = outs[0]
            pred_loc = outs[1]
            tgt_lbl = outs[2]
            tgt_bbox = outs[3]
            bbox_inside_weight = outs[4]

            self.assertIsNotNone(pred_scores)
            self.assertIsNotNone(pred_loc)
            self.assertIsNotNone(tgt_lbl)
            self.assertIsNotNone(tgt_bbox)
            self.assertIsNotNone(bbox_inside_weight)
            assert pred_scores.shape[1] == 1
            assert pred_loc.shape[1] == 4
            assert pred_loc.shape[1] == tgt_bbox.shape[1]
            print(str(program))


class TestYoloDetection(unittest.TestCase):
    def test_yolov3_loss(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            gt_box = layers.data(name='gt_box', shape=[10, 4], dtype='float32')
            gt_label = layers.data(name='gt_label', shape=[10], dtype='int32')
            gt_score = layers.data(name='gt_score', shape=[10], dtype='float32')
            loss = layers.yolov3_loss(
                x,
                gt_box,
                gt_label,
                [10, 13, 30, 13],
                [0, 1],
                10,
                0.7,
                32,
                gt_score=gt_score,
                use_label_smooth=False,
            )

            self.assertIsNotNone(loss)

    def test_yolo_box(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            img_size = layers.data(name='img_size', shape=[2], dtype='int32')
            boxes, scores = layers.yolo_box(
                x, img_size, [10, 13, 30, 13], 10, 0.01, 32
            )
            self.assertIsNotNone(boxes)
            self.assertIsNotNone(scores)

    def test_yolov3_loss_with_scale(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            gt_box = layers.data(name='gt_box', shape=[10, 4], dtype='float32')
            gt_label = layers.data(name='gt_label', shape=[10], dtype='int32')
            gt_score = layers.data(name='gt_score', shape=[10], dtype='float32')
            loss = layers.yolov3_loss(
                x,
                gt_box,
                gt_label,
                [10, 13, 30, 13],
                [0, 1],
                10,
                0.7,
                32,
                gt_score=gt_score,
                use_label_smooth=False,
                scale_x_y=1.2,
            )

            self.assertIsNotNone(loss)

    def test_yolo_box_with_scale(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            img_size = layers.data(name='img_size', shape=[2], dtype='int32')
            boxes, scores = layers.yolo_box(
                x, img_size, [10, 13, 30, 13], 10, 0.01, 32, scale_x_y=1.2
            )
            self.assertIsNotNone(boxes)
            self.assertIsNotNone(scores)


class TestMulticlassNMS2(unittest.TestCase):
    def test_multiclass_nms2(self):
        program = Program()
        with program_guard(program):
            bboxes = layers.data(
                name='bboxes', shape=[-1, 10, 4], dtype='float32'
            )
            scores = layers.data(name='scores', shape=[-1, 10], dtype='float32')
            output = fluid.contrib.multiclass_nms2(
                bboxes, scores, 0.3, 400, 200, 0.7
            )
            output2, index = fluid.contrib.multiclass_nms2(
                bboxes, scores, 0.3, 400, 200, 0.7, return_index=True
            )
            self.assertIsNotNone(output)
            self.assertIsNotNone(output2)
            self.assertIsNotNone(index)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
