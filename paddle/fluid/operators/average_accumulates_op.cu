/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/average_accumulates_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace operators {
template <>
void GetAccumulators<paddle::platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx, int64_t* num_updates_,
    int64_t* num_accumulates_, int64_t* old_num_accumulates_) {
  auto* in_old_num_accumulates = ctx.Input<Tensor>("in_old_num_accumulates");
  auto* in_num_accumulates = ctx.Input<Tensor>("in_num_accumulates");
  auto* in_num_updates = ctx.Input<Tensor>("in_num_updates");
  auto stream = ctx.cuda_device_context().stream();
  auto cuda_place =
      BOOST_GET_CONST(platform::CUDAPlace, in_old_num_accumulates->place());
  memory::Copy(platform::CPUPlace(), old_num_accumulates_, cuda_place,
               in_old_num_accumulates->data<int64_t>(), sizeof(int64_t),
               stream);
  memory::Copy(platform::CPUPlace(), num_accumulates_, cuda_place,
               in_num_accumulates->data<int64_t>(), sizeof(int64_t), stream);
  memory::Copy(platform::CPUPlace(), num_updates_, cuda_place,
               in_num_updates->data<int64_t>(), sizeof(int64_t), stream);
}

template <>
void SetAccumulators<paddle::platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx, int64_t num_updates_,
    int64_t num_accumulates_, int64_t old_num_accumulates_) {
  auto stream = ctx.cuda_device_context().stream();
  auto* out_old_num_accumulates = ctx.Output<Tensor>("out_old_num_accumulates");
  auto* out_num_accumulates = ctx.Output<Tensor>("out_num_accumulates");
  auto* out_num_updates = ctx.Output<Tensor>("out_num_updates");
  auto cuda_place =
      BOOST_GET_CONST(platform::CUDAPlace, out_old_num_accumulates->place());

  memory::Copy(cuda_place, out_old_num_accumulates->data<int64_t>(),
               platform::CPUPlace(), &old_num_accumulates_, sizeof(int64_t),
               stream);
  memory::Copy(cuda_place, out_num_accumulates->data<int64_t>(),
               platform::CPUPlace(), &num_accumulates_, sizeof(int64_t),
               stream);
  memory::Copy(cuda_place, out_num_updates->data<int64_t>(),
               platform::CPUPlace(), &num_updates_, sizeof(int64_t), stream);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    average_accumulates,
    ops::AverageAccumulatesKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AverageAccumulatesKernel<paddle::platform::CUDADeviceContext, double>);
