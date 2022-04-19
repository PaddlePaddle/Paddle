// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/elementwise/mkldnn/elementwise_mkldnn_op.h"

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_add>)

REGISTER_OP_KERNEL(
    elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNGradKernel<paddle::platform::bfloat16,
                                 dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNGradKernel<float, dnnl::algorithm::binary_add>)
