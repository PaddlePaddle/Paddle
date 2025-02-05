# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import paddle
from paddle.base import core
from paddle.base.layer_helper import LayerHelper

paddle.enable_static()


def multiclass_nms(
    bboxes,
    scores,
    score_threshold,
    nms_top_k,
    keep_top_k,
    nms_threshold=0.3,
    normalized=True,
    nms_eta=1.0,
    background_label=-1,
):
    helper = LayerHelper('multiclass_nms3', **locals())
    output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
    index = helper.create_variable_for_type_inference(dtype='int32')
    nms_rois_num = helper.create_variable_for_type_inference(dtype='int32')
    inputs = {'BBoxes': bboxes, 'Scores': scores}
    outputs = {'Out': output, 'Index': index, 'NmsRoisNum': nms_rois_num}

    helper.append_op(
        type="multiclass_nms3",
        inputs=inputs,
        attrs={
            'background_label': background_label,
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'nms_threshold': nms_threshold,
            'keep_top_k': keep_top_k,
            'nms_eta': nms_eta,
            'normalized': normalized,
        },
        outputs=outputs,
    )
    output.stop_gradient = True
    index.stop_gradient = True

    return output, index, nms_rois_num


class TestYoloBoxPass(unittest.TestCase):
    def test_yolo_box_pass(self):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            im_shape = paddle.static.data("im_shape", [1, 2])
            im_scale = paddle.static.data("im_scale", [1, 2])
            yolo_box0_x = paddle.static.data("yolo_box0_x", [1, 255, 19, 19])
            yolo_box1_x = paddle.static.data("yolo_box1_x", [1, 255, 38, 38])
            yolo_box2_x = paddle.static.data("yolo_box2_x", [1, 255, 76, 76])
            div = paddle.divide(im_shape, im_scale)
            cast = paddle.cast(div, "int32")
            boxes0, scores0 = paddle.vision.ops.yolo_box(
                yolo_box0_x, cast, [116, 90, 156, 198, 373, 326], 80, 0.005, 32
            )
            boxes1, scores1 = paddle.vision.ops.yolo_box(
                yolo_box1_x, cast, [30, 61, 62, 45, 59, 119], 80, 0.005, 16
            )
            boxes2, scores2 = paddle.vision.ops.yolo_box(
                yolo_box2_x, cast, [10, 13, 16, 30, 33, 23], 80, 0.005, 8
            )
            transpose0 = paddle.transpose(scores0, [0, 2, 1])
            transpose1 = paddle.transpose(scores1, [0, 2, 1])
            transpose2 = paddle.transpose(scores2, [0, 2, 1])
            concat0 = paddle.concat([boxes0, boxes1, boxes2], 1)
            concat1 = paddle.concat([transpose0, transpose1, transpose2], 2)
            out0, out1, out2 = multiclass_nms(
                concat0, concat1, 0.01, 1000, 100, 0.45, True, 1.0, 80
            )
        graph = core.Graph(program.desc)
        core.get_pass("yolo_box_fuse_pass").apply(graph)
        graph = paddle.base.framework.IrGraph(graph)
        op_nodes = graph.all_op_nodes()
        for op_node in op_nodes:
            op_type = op_node.op().type()
            self.assertTrue(op_type in ["yolo_box_head", "yolo_box_post"])


if __name__ == '__main__':
    unittest.main()
