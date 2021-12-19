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

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
struct GeluGradYFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T x, T dout) {
    // bool _approximate = false;
    // if(_approximate){
    MT mx = static_cast<MT>(x);
    MT tanh_out =
        tanh(static_cast<MT>(0.79788456) * mx *
             (static_cast<MT>(1) + static_cast<MT>(0.044715) * mx * mx));
    MT ans = static_cast<MT>(0.5) * mx *
                 ((static_cast<MT>(1) - tanh_out * tanh_out) *
                  (static_cast<MT>(0.79788456) +
                   static_cast<MT>(0.1070322243) * mx * mx)) +
             static_cast<MT>(0.5) * (static_cast<MT>(1) + tanh_out);
    return static_cast<T>(ans);
  }
};

template <typename T>
struct GeluGradXFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T x, T dout) {
    MT mx = static_cast<MT>(x);
    auto first =
        static_cast<MT>(0.5) *
        (erf(static_cast<MT>(1) + ((mx * static_cast<MT>(M_SQRT1_2)))));

    auto second = static_cast<MT>(0.5 * static_cast<MT>(M_2_SQRTPI) *
                                  static_cast<MT>(M_SQRT1_2)) *
                  mx * exp(-static_cast<MT>(0.5) * mx * mx);
    return static_cast<T>(static_cast<MT>(dout) * (first + second));
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
default_gelu_bw(const framework::ExecutionContext& ctx,
                const framework::Tensor* in, const framework::Tensor* dout,
                const bool approximate, framework::Tensor* out) {
  std::vector<const framework::Tensor*> ins;
  std::vector<framework::Tensor*> outs;
  ins = {in, dout};
  outs = {out};
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  if (approximate) {
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluGradYFunctor<T>());
  } else {
    // GeluGradXFunctor<T> gelu_grad_x_fun(approximate);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluGradXFunctor<T>());
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
