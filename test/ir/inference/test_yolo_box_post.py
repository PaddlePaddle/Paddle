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

import numpy as np

import paddle
from paddle.base.layer_helper import LayerHelper

paddle.enable_static()


def yolo_box_post(
    box0,
    box1,
    box2,
    im_shape,
    im_scale,
    anchors0=[116, 90, 156, 198, 373, 326],
    anchors1=[30, 61, 62, 45, 59, 119],
    anchors2=[10, 13, 16, 30, 33, 23],
    class_num=80,
    conf_thresh=0.005,
    downsample_ratio0=32,
    downsample_ratio1=16,
    downsample_ratio2=8,
    clip_bbox=True,
    scale_x_y=1.0,
    nms_threshold=0.45,
):
    helper = LayerHelper('yolo_box_post', **locals())
    output = helper.create_variable_for_type_inference(dtype=box0.dtype)
    nms_rois_num = helper.create_variable_for_type_inference(dtype='int32')
    inputs = {
        'Boxes0': box0,
        'Boxes1': box1,
        'Boxes2': box2,
        "ImageShape": im_shape,
        "ImageScale": im_scale,
    }
    outputs = {'Out': output, 'NmsRoisNum': nms_rois_num}

    helper.append_op(
        type="yolo_box_post",
        inputs=inputs,
        attrs={
            'anchors0': anchors0,
            'anchors1': anchors1,
            'anchors2': anchors2,
            'class_num': class_num,
            'conf_thresh': conf_thresh,
            'downsample_ratio0': downsample_ratio0,
            'downsample_ratio1': downsample_ratio1,
            'downsample_ratio2': downsample_ratio2,
            'clip_bbox': clip_bbox,
            'scale_x_y': scale_x_y,
            'nms_threshold': nms_threshold,
        },
        outputs=outputs,
    )
    output.stop_gradient = True
    nms_rois_num.stop_gradient = True
    return output, nms_rois_num


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "only support cuda kernel."
)
class TestYoloBoxPost(unittest.TestCase):
    def test_yolo_box_post(self):
        place = paddle.CUDAPlace(0)
        with paddle.pir_utils.OldIrGuard():
            program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(program, startup_program):
                box0 = paddle.static.data("box0", [1, 255, 19, 19])
                box1 = paddle.static.data("box1", [1, 255, 38, 38])
                box2 = paddle.static.data("box2", [1, 255, 76, 76])
                im_shape = paddle.static.data("im_shape", [1, 2])
                im_scale = paddle.static.data("im_scale", [1, 2])
                out, rois_num = yolo_box_post(
                    box0, box1, box2, im_shape, im_scale
                )
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            feed = {
                "box0": np.random.uniform(size=[1, 255, 19, 19]).astype(
                    "float32"
                ),
                "box1": np.random.uniform(size=[1, 255, 38, 38]).astype(
                    "float32"
                ),
                "box2": np.random.uniform(size=[1, 255, 76, 76]).astype(
                    "float32"
                ),
                "im_shape": np.array([[608.0, 608.0]], "float32"),
                "im_scale": np.array([[1.0, 1.0]], "float32"),
            }
            outs = exe.run(program, feed=feed, fetch_list=[out, rois_num])


if __name__ == '__main__':
    unittest.main()
