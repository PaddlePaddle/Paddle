/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dot_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
class SparseAttentionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::NotFound("This kernel only runs on GPU device."));

    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle
