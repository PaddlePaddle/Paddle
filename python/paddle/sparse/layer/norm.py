# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class BatchNorm(paddle.fluid.dygraph.BatchNorm):
    def __init__(self,
                 num_channels,
                 act=None,
                 is_test=False,
                 momentum=0.9,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32',
                 data_layout='NCHW',
                 in_place=False,
                 moving_mean_name=None,
                 moving_variance_name=None,
                 do_model_average_for_mean_and_var=True,
                 use_global_stats=False,
                 trainable_statistics=False):
        super(BatchNorm, self).__init__(
            num_channels,
            act=act,
            is_test=is_test,
            momentum=momentum,
            epsilon=epsilon,
            param_attr=param_attr,
            bias_attr=bias_attr,
            dtype=dtype,
            data_layout,
            in_place=in_place,
            moving_mean_name=moving_mean_name,
            moving_variance_name=moving_variance_name,
            do_model_average_for_mean_and_var=do_model_average_for_mean_and_var,
            use_global_stats=use_global_stats,
            trainable_statistics=tranable_statistics)

        def forward(self, input):
            values = input.values()
            out = super(BatchNorm, self).forward(values)
            return paddle.sparse.sparse_coo_tensor(
                input.indices(),
                out,
                shape=input.shape,
                stop_gradient=input.stop_gradient)
