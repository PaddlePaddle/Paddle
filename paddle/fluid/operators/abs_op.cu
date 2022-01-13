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

#include "paddle/fluid/operators/abs_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename Enable = void>
struct CudaAbsFunctor;

template <typename T>
struct CudaAbsFunctor<T, math::Complex<T, math::Real<T>>> {
  __device__ __forceinline__ math::Real<T> operator()(const T x) const {
    return abs(x);
  }
};

template <typename T>
struct CudaAbsFunctor<T, math::NoComplex<T, math::Real<T>>> {
  __device__ __forceinline__ T operator()(const T x) const {
    return std::abs(x);
  }
};

template <typename T>
class AbsKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<math::Real<T>>(context.GetPlace());

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    auto functor = CudaAbsFunctor<T>();
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<
        ElementwiseType::kUnary, T, math::Real<T>>(dev_ctx, ins, &outs,
                                                   functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    abs, ops::AbsKernel<plat::CUDADeviceContext, float>,
    ops::AbsKernel<plat::CUDADeviceContext, double>,
    ops::AbsKernel<plat::CUDADeviceContext, int>,
    ops::AbsKernel<plat::CUDADeviceContext, int64_t>,
    ops::AbsKernel<plat::CUDADeviceContext, plat::float16>,
    ops::AbsKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::AbsKernel<plat::CUDADeviceContext, plat::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    abs_grad, ops::AbsGradKernel<plat::CUDADeviceContext, float>,
    ops::AbsGradKernel<plat::CUDADeviceContext, double>,
    ops::AbsGradKernel<plat::CUDADeviceContext, int>,
    ops::AbsGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::AbsGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::AbsGradKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::AbsGradKernel<plat::CUDADeviceContext, plat::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    abs_grad_grad, ops::AbsDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::AbsDoubleGradKernel<plat::CUDADeviceContext, plat::complex<double>>);
