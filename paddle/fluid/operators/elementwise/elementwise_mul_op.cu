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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMulKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_EQ(x_var != nullptr, true,
                      platform::errors::InvalidArgument(
                          "Cannot get input Variable X, Variable name = %s.",
                          ctx.InputName("X")));
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    if (x_var->IsType<phi::SelectedRows>()) {
      framework::Tensor x_for_selectedrows;
      std::vector<const framework::Tensor*> ins;
      std::vector<framework::Tensor*> outs;
      int axis =
          PackTensorsIntoVector<T>(ctx, &ins, &outs, &x_for_selectedrows);
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          cuda_ctx, ins, &outs, axis, MulFunctor<T>());
    } else if (x_var->IsType<framework::LoDTensor>()) {
      auto* x_lod = ctx.Input<framework::LoDTensor>("X");
      auto* y_lod = ctx.Input<framework::LoDTensor>("Y");
      auto* z_lod = ctx.Output<framework::LoDTensor>("Out");
      z_lod->mutable_data<T>(ctx.GetPlace());

      int axis = ctx.Attr<int>("axis");
      auto pt_x = paddle::experimental::MakePhiDenseTensor(*x_lod);
      auto pt_y = paddle::experimental::MakePhiDenseTensor(*y_lod);
      auto pt_z = paddle::experimental::MakePhiDenseTensor(*z_lod);
      phi::MultiplyRawKernel<T>(static_cast<const phi::GPUContext&>(cuda_ctx),
                                *pt_x.get(), *pt_y.get(), axis, pt_z.get());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_mul, ops::ElementwiseMulKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::bfloat16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<double>>);
