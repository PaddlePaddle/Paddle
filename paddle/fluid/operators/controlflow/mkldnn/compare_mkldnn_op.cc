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

#define REGISTER_COMPARE_OP_MKLDNN_KERNEL(op_type, algo)                        \
  REGISTER_OP_KERNEL(                                                           \
      op_type,                                                                  \
      MKLDNN,                                                                   \
      ::paddle::platform::CPUPlace,                                             \
      ops::EltwiseMKLDNNKernel<float, algo>);                                   \

REGISTER_COMPARE_OP_MKLDNN_KERNEL(equal, dnnl::algorithm::binary_eq)
REGISTER_COMPARE_OP_MKLDNN_KERNEL(not_equal, dnnl::algorithm::binary_ne)
REGISTER_COMPARE_OP_MKLDNN_KERNEL(greater_than, dnnl::algorithm::binary_gt)
REGISTER_COMPARE_OP_MKLDNN_KERNEL(greater_equal, dnnl::algorithm::binary_ge)
REGISTER_COMPARE_OP_MKLDNN_KERNEL(less_than, dnnl::algorithm::binary_lt)
REGISTER_COMPARE_OP_MKLDNN_KERNEL(less_equal, dnnl::algorithm::binary_le)
