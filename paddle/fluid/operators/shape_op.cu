/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda.h>
#include "paddle/fluid/operators/shape_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

template <typename T>
class ShapeCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_var = ctx.InputVar("Input");
    framework::DDim in_dims;
    if (in_var->IsType<SelectedRows>()) {
      in_dims = in_var->Get<SelectedRows>().value().dims();
    } else {
      in_dims = in_var->Get<LoDTensor>().dims();
    }

    auto* out_t = ctx.Output<Tensor>("Out");
    auto dim_size = in_dims.size();
    out_t->Resize({dim_size});
    auto* out_data = out_t->mutable_data<int64_t>(ctx.GetPlace());
    auto stream = ctx.cuda_device_context().stream();
    auto target_gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    VLOG(1) << "data of tensor.shape: " << in_dims;
    // copy in_dims_data from CPU to GPU
    memory::Copy(target_gpu_place, out_data, platform::CPUPlace(),
                 in_dims.Get(), dim_size * sizeof(int64_t), stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    shape, paddle::operators::ShapeCUDAKernel<int>,
    paddle::operators::ShapeCUDAKernel<int32_t>,
    paddle::operators::ShapeCUDAKernel<int64_t>,
    paddle::operators::ShapeCUDAKernel<float>,
    paddle::operators::ShapeCUDAKernel<double>,
    paddle::operators::ShapeCUDAKernel<paddle::platform::float16>);
