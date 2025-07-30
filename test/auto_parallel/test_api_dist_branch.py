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

import unittest

import numpy as np

import paddle
import paddle.distributed as dist


# For API generation which have different type of DistTensor Input and Output
class TestDygraphAPIForDistTensorBranch(unittest.TestCase):
    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05)

    def create_local_and_dist_tensor_pair(self, np_array):
        if np_array.dtype == np.float32:
            local_t = paddle.to_tensor(np_array, dtype='float32')
        elif np_array.dtype == np.float16:
            local_t = paddle.to_tensor(np_array, dtype='float16')
        elif np_array.dtype == np.int32:
            local_t = paddle.to_tensor(np_array, dtype='int32')
        elif np_array.dtype == np.bool_:
            local_t = paddle.to_tensor(np_array, dtype='bool')

        mesh = dist.ProcessMesh([0], dim_names=["x"])
        dist_t = dist.shard_tensor(np_array, mesh, [dist.Replicate()])

        local_t.stop_gradient = False
        dist_t.stop_gradient = False

        return local_t, dist_t

    def create_local_and_dist_tensor_list_pair(self, np_array_list):
        assert isinstance(
            np_array_list, list
        ), "input should be list of np_array!"
        local_t_list = []
        dist_t_list = []
        for np_array in np_array_list:
            local_t, dist_t = self.create_local_and_dist_tensor_pair(np_array)
            local_t_list.append(local_t)
            dist_t_list.append(dist_t)
        return local_t_list, dist_t_list

    def create_two_local_tensor_pair(self, np_array):
        if np_array.dtype == np.float32:
            local_t_1 = paddle.to_tensor(np_array, dtype='float32')
            local_t_2 = paddle.to_tensor(np_array, dtype='float32')
        elif np_array.dtype == np.float16:
            local_t_1 = paddle.to_tensor(np_array, dtype='float16')
            local_t_2 = paddle.to_tensor(np_array, dtype='float16')
        elif np_array.dtype == np.int32:
            local_t_1 = paddle.to_tensor(np_array, dtype='int32')
            local_t_2 = paddle.to_tensor(np_array, dtype='int32')
        elif np_array.dtype == np.bool_:
            local_t_1 = paddle.to_tensor(np_array, dtype='bool')
            local_t_2 = paddle.to_tensor(np_array, dtype='bool')

        local_t_1.stop_gradient = False
        local_t_2.stop_gradient = False

        return local_t_1, local_t_2

    # mixed type of inputs: DenseTensor and DistTensor
    def test_matmul_api_for_mixed_inputs_type(self):
        x = np.random.random(size=[4, 4]).astype("float32")
        y = np.random.random(size=[4, 4]).astype("float32")
        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_y_1, local_y_2 = self.create_two_local_tensor_pair(y)
        local_out = paddle.matmul(local_x, local_y_1)
        dist_out = paddle.matmul(dist_x, local_y_2)
        self.check_tensor_eq(local_out, dist_out)

        # test backward
        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_x.grad, dist_x.grad)
        self.check_tensor_eq(local_y_1.grad, local_y_2.grad)

    # input: std::vector<phi::Tensor>
    # output: phi::Tensor
    def test_concat_for_dist_tensor(self):
        x1 = np.random.random(size=[4, 4]).astype("float32")
        x2 = np.random.random(size=[4, 4]).astype("float32")
        x3 = np.random.random(size=[4, 4]).astype("float32")
        local_in1, dist_in1 = self.create_local_and_dist_tensor_pair(x1)
        local_in2, dist_in2 = self.create_local_and_dist_tensor_pair(x2)
        local_in3, dist_in3 = self.create_local_and_dist_tensor_pair(x3)
        local_out = paddle.concat([local_in1, local_in2, local_in3])
        dist_out = paddle.concat([dist_in1, dist_in2, dist_in3])
        self.check_tensor_eq(local_out, dist_out)
        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in1.grad, dist_in1.grad)
        self.check_tensor_eq(local_in2.grad, dist_in2.grad)
        self.check_tensor_eq(local_in3.grad, dist_in3.grad)

    # TODO(GhostScreaming): Support paddle.concat backward later.
    # input: std::vector<phi::Tensor>
    # output: std::vector<phi::Tensor>
    def test_broadcast_tensors_for_dist_tensor(self):
        x1 = np.random.random(size=[4, 4]).astype("float32")
        x2 = np.random.random(size=[4, 4]).astype("float32")
        local_in1, dist_in1 = self.create_local_and_dist_tensor_pair(x1)
        local_in2, dist_in2 = self.create_local_and_dist_tensor_pair(x2)

        local_out1, local_out2 = paddle.broadcast_tensors(
            [local_in1, local_in2]
        )
        dist_out1, dist_out2 = paddle.broadcast_tensors([dist_in1, dist_in2])
        self.check_tensor_eq(local_out1, dist_out1)
        self.check_tensor_eq(local_out2, dist_out2)

        local_out = paddle.concat([local_out1, local_out2])
        dist_out = paddle.concat([dist_out1, dist_out2])

        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in1.grad, dist_in1.grad)
        self.check_tensor_eq(local_in2.grad, dist_in2.grad)

    # input: paddle::optional<phi::Tensor>
    # output: phi::Tensor
    def test_bincount_api_for_dist_tensor(self):
        x = np.random.random(size=[16]).astype("int32")
        weight = np.random.random(size=[16]).astype("float32")
        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_weight, dist_weight = self.create_local_and_dist_tensor_pair(
            weight
        )

        local_out = paddle.bincount(local_x, weights=local_weight)
        dist_out = paddle.bincount(dist_x, weights=dist_weight)

        self.check_tensor_eq(local_out, dist_out)

    # input: paddle::optional<std::vector<phi::Tensor>>
    # output: phi::Tensor
    def test_linear_interp_for_dist_tensor(self):
        out_size = np.array(
            [
                50,
            ]
        ).astype("int32")
        shape = [1, 3, 100]
        size1 = np.array([50]).astype("int32")
        scale = 0.5
        scale_list = []
        for _ in range(len(shape) - 2):
            scale_list.append(scale)
        scale = list(map(float, scale_list))

        x = np.random.random(size=shape).astype("float32")
        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_out_size, dist_out_size = self.create_local_and_dist_tensor_pair(
            out_size
        )
        local_size1, dist_size1 = self.create_local_and_dist_tensor_pair(size1)

        local_scale, dist_scale = self.create_local_and_dist_tensor_pair(
            np.array([0.5]).astype("float32")
        )
        local_out = paddle._C_ops.linear_interp(
            local_x,
            local_out_size,  # Outsize
            [local_size1],  # SizeTensor
            local_scale,  # Scale
            'NCHW',  # data_layout
            -1,  # out_d
            -1,  # out_h
            50,  # in_w * out_w
            scale,
            'linear',  # interp_method
            False,  # align_corners
            1,  # align_mode
        )
        dist_out = paddle._C_ops.linear_interp(
            dist_x,
            dist_out_size,  # Outsize
            [dist_size1],  # SizeTensor
            dist_scale,  # Scale
            'NCHW',  # data_layout
            -1,  # out_d
            -1,  # out_h
            50,  # in_w * out_w
            scale,
            'linear',
            False,  # align_corners
            1,  # align_mode
        )
        self.check_tensor_eq(local_out, dist_out)

    # input: std::vector<phi::Tensor>, phi::Tensor
    # output: inplace std::vector<phi::Tensor>, inplace phi::Tensor
    def test_check_finite_and_unscale_for_dist_tensor(self):
        x = np.random.random((1024, 1024)).astype("float32")
        x[128][128] = np.inf
        scale = np.random.random(1).astype("float32")
        found_inf = np.array([0]).astype(np.bool_)

        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_scale, dist_scale = self.create_local_and_dist_tensor_pair(scale)
        (
            local_found_inf,
            dist_found_inf,
        ) = self.create_local_and_dist_tensor_pair(found_inf)

        paddle._C_ops.check_finite_and_unscale_(
            [local_x],
            local_scale,
            [local_x],
            local_found_inf,
        )
        paddle._C_ops.check_finite_and_unscale_(
            [dist_x],
            dist_scale,
            [dist_x],
            dist_found_inf,
        )
        self.check_tensor_eq(local_x, dist_x)
        self.check_tensor_eq(local_found_inf, dist_found_inf)

    # multi kernel functions
    def test_adagrad_for_dist_tensor(self):
        dtype = np.float16
        mp_dtype = np.float32
        shape = [123, 321]

        param = np.random.random(shape).astype(dtype)
        grad = np.random.random(shape).astype(dtype)
        moment = np.random.random(shape).astype(mp_dtype)
        master_param = param.astype(mp_dtype)

        lr = np.array([0.002]).astype("float32")
        epsilon = 1e-8

        local_param, dist_param = self.create_local_and_dist_tensor_pair(param)
        local_grad, dist_grad = self.create_local_and_dist_tensor_pair(grad)
        local_lr, dist_lr = self.create_local_and_dist_tensor_pair(lr)
        local_moment, dist_moment = self.create_local_and_dist_tensor_pair(
            moment
        )
        (
            local_master_param,
            dist_master_param,
        ) = self.create_local_and_dist_tensor_pair(master_param)
        (
            local_param_out,
            local_moment_out,
            local_master_param_out,
        ) = paddle._C_ops.adagrad_(
            local_param,
            local_grad,
            local_moment,
            local_lr,
            local_master_param,
            epsilon,
            True,
        )

        (
            dist_param_out,
            dist_moment_out,
            dist_master_param_out,
        ) = paddle._C_ops.adagrad_(
            dist_param,
            dist_grad,
            dist_moment,
            dist_lr,
            dist_master_param,
            epsilon,
            True,
        )

        self.check_tensor_eq(local_param_out, dist_param_out)
        self.check_tensor_eq(local_moment_out, dist_moment_out)
        self.check_tensor_eq(local_master_param_out, dist_master_param_out)

    # input: std::vector<phi::Tensor>, phi::Tensor
    # output: inplace paddle::optional<std::vector<phi::Tensor>>, inplace phi::Tensor
    def test_merged_adam_for_dist_tensor(self):
        dtype = np.float16
        mp_dtype = np.float32
        lr_shape = [[1], [1], [1], [1]]
        shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]

        epsilon = 0.9
        beta1 = 0.9
        beta2 = 0.99
        params = [np.random.random(s).astype(dtype) for s in shapes]
        grads = [np.random.random(s).astype(dtype) for s in shapes]
        lrs = [np.random.random(s).astype(mp_dtype) for s in lr_shape]
        moment1s = [np.random.random(s).astype(mp_dtype) for s in shapes]
        moment2s = [np.random.random(s).astype(mp_dtype) for s in shapes]
        moment2s_max = [np.zeros(s).astype(mp_dtype) for s in shapes]
        beta1_pows = [np.random.random(s).astype(mp_dtype) for s in lr_shape]
        beta2_pows = [np.random.random(s).astype(mp_dtype) for s in lr_shape]
        master_params = [p.astype(mp_dtype) for p in params]

        local_param, dist_param = self.create_local_and_dist_tensor_list_pair(
            params
        )
        local_grads, dist_grads = self.create_local_and_dist_tensor_list_pair(
            grads
        )
        local_lrs, dist_lrs = self.create_local_and_dist_tensor_list_pair(lrs)
        (
            local_moment1s,
            dist_moment1s,
        ) = self.create_local_and_dist_tensor_list_pair(moment1s)
        (
            local_moment2s,
            dist_moment2s,
        ) = self.create_local_and_dist_tensor_list_pair(moment2s)
        (
            local_moment2s_max,
            dist_moment2s_max,
        ) = self.create_local_and_dist_tensor_list_pair(moment2s_max)
        (
            local_beta1_pows,
            dist_beta1_pows,
        ) = self.create_local_and_dist_tensor_list_pair(beta1_pows)
        (
            local_beta2_pows,
            dist_beta2_pows,
        ) = self.create_local_and_dist_tensor_list_pair(beta2_pows)
        (
            local_master_params,
            dist_master_params,
        ) = self.create_local_and_dist_tensor_list_pair(master_params)

        (
            local_param_out,
            local_moment1s_out,
            local_moment2s_out,
            local_moment2s_max_out,
            local_beta1_pow_out,
            local_beta2_pow_out,
            local_master_param_out,
        ) = paddle._C_ops.merged_adam_(
            local_param,
            local_grads,
            local_lrs,
            local_moment1s,
            local_moment2s,
            local_moment2s_max,
            local_beta1_pows,
            local_beta2_pows,
            local_master_params,
            beta1,
            beta2,
            epsilon,
            True,
            False,
            False,
        )

        (
            dist_param_out,
            dist_moment1s_out,
            dist_moment2s_out,
            dist_moment2s_max_out,
            dist_beta1_pow_out,
            dist_beta2_pow_out,
            dist_master_param_out,
        ) = paddle._C_ops.merged_adam_(
            dist_param,
            dist_grads,
            dist_lrs,
            dist_moment1s,
            dist_moment2s,
            dist_moment2s_max,
            dist_beta1_pows,
            dist_beta2_pows,
            dist_master_params,
            beta1,
            beta2,
            epsilon,
            True,
            False,
            False,
        )
        for i in range(len(local_param_out)):
            self.check_tensor_eq(local_param_out[i], dist_param_out[i])
            self.check_tensor_eq(local_moment1s_out[i], dist_moment1s_out[i])
            self.check_tensor_eq(local_moment2s_out[i], dist_moment2s_out[i])
            self.check_tensor_eq(local_beta1_pow_out[i], dist_beta1_pow_out[i])
            self.check_tensor_eq(local_beta2_pow_out[i], dist_beta2_pow_out[i])
            self.check_tensor_eq(
                local_master_param_out[i], dist_master_param_out[i]
            )

    # intermediate dygraph api test
    def test_layer_norm_for_intermediate_dist_tensor(self):
        x = np.random.random((2, 3, 10, 10)).astype("float32")
        weight = np.random.random(300).astype("float32")
        bias = np.random.random(300).astype("float32")

        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_weight, dist_weight = self.create_local_and_dist_tensor_pair(
            weight
        )
        local_bias, dist_bias = self.create_local_and_dist_tensor_pair(bias)

        local_out = paddle.nn.functional.layer_norm(
            local_x,
            local_x.shape[1:],
            local_weight,
            local_bias,
        )
        dist_out = paddle.nn.functional.layer_norm(
            dist_x,
            dist_x.shape[1:],
            dist_weight,
            dist_bias,
        )
        self.check_tensor_eq(local_out, dist_out)


if __name__ == "__main__":
    unittest.main()
