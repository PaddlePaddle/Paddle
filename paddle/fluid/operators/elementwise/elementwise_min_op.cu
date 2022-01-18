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

#include "paddle/fluid/operators/elementwise/elementwise_min_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMinKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(dev_ctx, ins, &outs, axis,
                                                      MinFunctor<T>());
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
ElementwiseMinGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  const auto place = ctx.GetPlace();
  if (dx != nullptr && dy != nullptr) {
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    GetGradXAndYOut<ElementwiseType::kTernary, T>(
        dev_ctx, place, axis, ins, dout, dx, dy, MinGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    GetGradXOrYOut<ElementwiseType::kTernary, T>(
        dev_ctx, place, axis, ins, dout, dx, MinGradXFunctor<T>());
  } else if (dx == nullptr && dy != nullptr) {
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    GetGradXOrYOut<ElementwiseType::kTernary, T>(
        dev_ctx, place, axis, ins, dout, dy, MinGradYFunctor<T>());
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    elementwise_min,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);

REGISTER_OP_CUDA_KERNEL(
    elementwise_fmin,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext,
                               paddle::platform::float16>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_fmin_grad,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext,
                                   paddle::platform::float16>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
