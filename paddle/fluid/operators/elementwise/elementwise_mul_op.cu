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
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

// only can include the headers in paddle/top/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/math.h"
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
    if (x_var->IsType<framework::SelectedRows>()) {
      framework::Tensor x_for_selectedrows;
      std::vector<const framework::Tensor*> ins;
      std::vector<framework::Tensor*> outs;
      int axis =
          PackTensorsIntoVector<T>(ctx, &ins, &outs, &x_for_selectedrows);
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, MulFunctor<T>());
    } else if (x_var->IsType<framework::LoDTensor>()) {
      auto* x_lod = ctx.Input<framework::LoDTensor>("X");
      auto* y_lod = ctx.Input<framework::LoDTensor>("Y");
      auto* z_lod = ctx.Output<framework::LoDTensor>("Out");
      z_lod->mutable_data<T>(ctx.GetPlace());

      int axis = ctx.Attr<int>("axis");
      auto pt_x = paddle::experimental::MakePtenDenseTensor(*x_lod);
      auto pt_y = paddle::experimental::MakePtenDenseTensor(*y_lod);
      auto pt_z = paddle::experimental::MakePtenDenseTensor(*z_lod);
      pten::MultiplyKernel<T>(cuda_ctx, *pt_x.get(), *pt_y.get(), axis,
                              pt_z.get());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
ElementwiseMulGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  const auto place = ctx.GetPlace();

  if (dx != nullptr && dy != nullptr) {
    dx->mutable_data<T>(place);
    if (dx->IsSharedBufferWith(*dout)) {
      dx->clear();
      dx->mutable_data<T>(x->dims(), place);
    }
    std::vector<const framework::Tensor*> ins = {dout, y, x};
    GetGradXAndYOut<ElementwiseType::kBinary, T>(
        dev_ctx, place, axis, ins, dout, dx, dy, MulGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    dx->mutable_data<T>(place);
    if (dx->IsSharedBufferWith(*dout)) {
      dx->clear();
      dx->mutable_data<T>(x->dims(), place);
    }
    std::vector<const framework::Tensor*> ins = {dout, y};
    GetGradXOrYOut<ElementwiseType::kBinary, T>(dev_ctx, place, axis, ins, dout,
                                                dx, MulGradFunctor<T>());
  } else if (dx == nullptr && dy != nullptr) {
    std::vector<const framework::Tensor*> ins = {dout, x};
    GetGradXOrYOut<ElementwiseType::kBinary, T>(dev_ctx, place, axis, ins, dout,
                                                dy, MulGradFunctor<T>());
  }
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_mul, ops::ElementwiseMulKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<float>>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<float>>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_triple_grad,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<float>>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<double>>);
