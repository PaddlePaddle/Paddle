# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import pytest

import paddle


def _compare(cpu_res, gcu_res):
    assert len(cpu_res) == len(gcu_res)
    for i in range(len(cpu_res)):
        out = gcu_res[i]
        exp = cpu_res[i]
        assert (
            out.shape == exp.shape
        ), f"out shape: {out.shape}, expect: {exp.shape}"
        if exp.dtype in [np.float16, np.float32, np.float64]:
            np.testing.assert_allclose(
                out, exp, rtol=1.0e-6, atol=1.0e-5, equal_nan=False
            )
        elif exp.dtype in [
            bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]:
            assert np.all(out == exp)
        else:
            assert 0, 'unsupport data type'


def _calc_output(x, y, box_normalized=True):
    paddle.enable_static()
    main_program = paddle.static.Program()
    main_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program):
            block = main_program.global_block()
            data_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
            data_y = paddle.static.data(name='y', shape=y.shape, dtype=y.dtype)
            data_x.stop_gradient = False
            data_y.stop_gradient = False
            out = block.create_var(name='iou_similarity_out', dtype=x.dtype)
            op = block.append_op(
                type='iou_similarity',
                inputs={'X': data_x, 'Y': data_y},
                outputs={'Out': out},
                attrs={'box_normalized': box_normalized},
            )
            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)

    cpu_exe = paddle.static.Executor(paddle.CPUPlace())
    cpu_res = cpu_exe.run(
        main_program,
        feed={'x': x, 'y': y},
        fetch_list=['iou_similarity_out'],
        return_numpy=True,
    )

    gcu_exe = paddle.static.Executor('gcu')
    gcu_res = gcu_exe.run(
        main_program,
        feed={'x': x, 'y': y},
        fetch_list=['iou_similarity_out'],
        return_numpy=True,
    )

    print(f'PROG_DEBUG main_program:{main_program}')
    print(
        'cpu_res shape:{}, gcu_res shape:{}'.format(
            cpu_res[0].shape, gcu_res[0].shape
        )
    )
    print(f'PROG_DEBUG cpu_res:{cpu_res}')
    print(f'PROG_DEBUG gcu_res:{gcu_res}')

    return cpu_res, gcu_res


@pytest.mark.iou_similarity
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_iou_similarity():
    x = np.random.random(size=[5760, 4]).astype('float32')
    y = np.random.random(size=[2278, 4]).astype('float32')
    cpu_res, gcu_res = _calc_output(x, y)
    _compare(cpu_res, gcu_res)


test_iou_similarity()
