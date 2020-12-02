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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/transpose_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

bool XPUSupported(int ndims, const std::vector<int>& axis) {
  /*
   * XPU currently support:
   * permute = {0, 2, 1}, permute = {1, 0},
   * permute = {0, 2, 1, 3}, permute = {1, 0, 2},
   * permute = {0, 2, 3, 1}
   */
  bool is_supported = false;
  std::vector<int> permute_10(2, 0);
  std::vector<int> permute_102(3, 0);
  std::vector<int> permute_021(3, 0);
  std::vector<int> permute_210(3, 0);
  std::vector<int> permute_0213(4, 0);
  std::vector<int> permute_0231(4, 0);
  std::vector<int> permute_0312(4, 0);
  std::vector<int> permute_3201(4, 0);
  permute_10[0] = 1;
  permute_102[0] = 1;
  permute_102[2] = 2;
  permute_021[1] = 2;
  permute_021[2] = 1;
  permute_210[0] = 2;
  permute_210[1] = 1;
  permute_0213[1] = 2;
  permute_0213[2] = 1;
  permute_0213[3] = 3;
  permute_0231[1] = 2;
  permute_0231[2] = 3;
  permute_0231[3] = 1;
  permute_0312[1] = 3;
  permute_0312[2] = 1;
  permute_0312[3] = 2;
  permute_3201[0] = 3;
  permute_3201[1] = 2;
  permute_3201[3] = 1;
  switch (ndims) {
    case 2:
      if (axis == permute_10) {
        is_supported = true;
      }
      break;
    case 3:
      if ((axis == permute_021) || (axis == permute_102) ||
          (axis == permute_210)) {
        is_supported = true;
      }
      break;
    case 4:
      if ((axis == permute_0213) || (axis == permute_0231) ||
          (axis == permute_0312) || (axis == permute_3201)) {
        is_supported = true;
      }
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Tensors with rank only 2, 3 and 4 are supported on XPU"));
  }
  return is_supported;
}

template <typename DeviceContext, typename T>
class TransposeXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x = context.Input<framework::Tensor>("X");
    auto out = context.Output<framework::Tensor>("Out");
    // axis is permute
    auto axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    const auto x_dims = x->dims();

    const T* x_data = x->data<T>();
    T* y_data = out->mutable_data<T>(context.GetPlace());
    if (!XPUSupported(ndims, axis)) {
      VLOG(0) << "XPU does not support the permute, try to do on cpu";
      framework::Tensor x_cpu;
      framework::Tensor out_cpu;
      auto x_cpu_data = x_cpu.mutable_data<T>(x->dims(), platform::CPUPlace());
      auto out_cpu_data =
          out_cpu.mutable_data<T>(out->dims(), platform::CPUPlace());
      memory::Copy(platform::CPUPlace(), reinterpret_cast<void*>(x_cpu_data),
                   BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                   (const void*)x_data, x->numel() * sizeof(T));

      const platform::CPUDeviceContext* cpu_dev_ctx =
          static_cast<const platform::CPUDeviceContext*>(
              platform::DeviceContextPool::Instance().Get(
                  platform::CPUPlace()));
      TransCompute<platform::CPUDeviceContext, T>(ndims, *cpu_dev_ctx, x_cpu,
                                                  &out_cpu, axis);
      memory::Copy(BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                   reinterpret_cast<void*>(y_data), platform::CPUPlace(),
                   (const void*)out_cpu_data, out->numel() * sizeof(T));
      return;
    }

    std::vector<int> x_shape_host(ndims, 0);
    for (int i = 0; i < ndims; ++i) {
      x_shape_host[i] = x_dims[i];
    }
    int* permute_host = axis.data();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::transpose(dev_ctx.x_context(), x_data, y_data,
                           x_shape_host.data(), permute_host, ndims);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error! error code=%d", r));
  }
};

template <typename DeviceContext, typename T>
class TransposeGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    x_grad->mutable_data<T>(context.GetPlace());
    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    int ndims = axis.size();
    if (!XPUSupported(ndims, reversed_axis)) {
      PADDLE_THROW(
          platform::errors::Unimplemented("XPU does not support the permute"));
    }

    std::vector<int> out_shape_host(ndims, 0);
    for (int i = 0; i < ndims; ++i) {
      out_shape_host[i] = out_grad->dims()[i];
    }
    int* permute_host = reversed_axis.data();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::transpose(dev_ctx.x_context(), out_grad->data<T>(),
                           x_grad->data<T>(), out_shape_host.data(),
                           permute_host, ndims);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error! error code=%d", r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    transpose,
    ops::TransposeXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose_grad,
    ops::TransposeGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose2,
    ops::TransposeXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose2_grad,
    ops::TransposeGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
