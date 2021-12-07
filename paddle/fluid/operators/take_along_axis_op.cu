/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/take_along_axis_op.h"

namespace paddle {
namespace operators {

template <typename T>
class TakeAlongAxisCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto input = ctx.Input<Tensor>("Input");
    auto dim = ctx.Attr<int>("Dim");
    auto index = ctx.Input<Tensor>("Index");
    auto result = ctx.Output<Tensor>("Result");
    result->Resize(index->dims());
    result->mutable_data<T>(ctx.GetPlace());

    //     resize_out(result, index.sizes());
    //     check_no_internal_overlap(self, result);
    //     check_no_partial_overlap(result, index);

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      gpu_gather_kernel<T, int32_t>(*input, dim, *index, *result);
    } else if (index_type == framework::proto::VarType::INT64) {
      gpu_gather_kernel<T, int64_t>(*input, dim, *index, *result);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(take_along_axis, ops::TakeAlongAxisCUDAKernel<float>,
                        ops::TakeAlongAxisCUDAKernel<double>,
                        ops::TakeAlongAxisCUDAKernel<int64_t>,
                        ops::TakeAlongAxisCUDAKernel<int>,
                        ops::TakeAlongAxisCUDAKernel<plat::float16>);
// REGISTER_OP_CUDA_KERNEL(gather_grad, ops::GatherGradOpCUDAKernel<float>,
//                         ops::GatherGradOpCUDAKernel<double>,
//                         ops::GatherGradOpCUDAKernel<int64_t>,
//                         ops::GatherGradOpCUDAKernel<int>,
//                         ops::GatherGradOpCUDAKernel<plat::float16>);
