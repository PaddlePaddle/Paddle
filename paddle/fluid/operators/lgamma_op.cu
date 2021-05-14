// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <unsupported/Eigen/SpecialFunctions>
#include "paddle/fluid/operators/lgamma_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct LgammaGradFunctorCUDA {
  LgammaGradFunctorCUDA(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = dout_[idx] / Eigen::numext::digamma(x_[idx]);
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <typename T>
class LgammaGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<T>();
    auto* x_data = x->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    auto& dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, numel);
    LgammaGradFunctorCUDA<T> functor(dout_data, x_data, dx_data, numel);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    lgamma, ops::LgammaKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LgammaKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    lgamma_grad,
    ops::LgammaGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LgammaGradKernel<paddle::platform::CUDADeviceContext, double>);
