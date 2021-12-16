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

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
struct GeluXFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T x) {
    MT mx = static_cast<MT>(x);
    MT temp = erf(mx * static_cast<MT>(M_SQRT1_2));
    MT out = mx * static_cast<MT>(0.5) * (static_cast<MT>(1) + temp);
    return static_cast<T>(out);
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
default_gelu_fw(const framework::ExecutionContext& ctx,
                const framework::Tensor* in, const bool approximate,
                framework::Tensor* out) {
  std::vector<const framework::Tensor*> ins;
  std::vector<framework::Tensor*> outs;
  ins = {in};
  outs = {out};
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  if (approximate) {
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, paddle::operators::math::GeluFunctor<T>());
  } else {
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluXFunctor<T>());
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gelu, ops::GeluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    gelu_grad, ops::GeluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
