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

#include <unsupported/Eigen/SpecialFunctions>
#include "paddle/fluid/operators/digamma_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/math/complex_functors.h"

namespace paddle {
namespace operators {

template <typename T, typename Enable = void>
struct CudaDigammaFunctor;

template <typename T>
struct CudaDigammaFunctor<T, math::NoComplex<T, math::Real<T>>> {
  __device__ __forceinline__ T operator()(const T* args) const {
    return Eigen::numext::digamma(args[0]);
  }
};

template <typename T>
class DigammaKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<math::Real<T>>(context.GetPlace());

    auto& dev_ctx = context.device_context<platform::CUDADeviceContext>();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    auto functor = CudaDigammaFunctor<T>();
    LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary, T,
                                        math::Real<T>>(dev_ctx, ins, &outs,
                                                       functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    digamma, ops::DigammaKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DigammaKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    digamma_grad,
    ops::DigammaGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DigammaGradKernel<paddle::platform::CUDADeviceContext, double>);
