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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// CPU Kernel ： out = x + y
template <typename DeviceContext,
          typename T>  //
class AddEqualDimKernel : public framework::OpKernel<T> {
 public:
  // 重写OpKernel::Compute()
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "AddEqualDim_kernel_start\n";
    // 输入、输出
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* out = context.Output<Tensor>("Out");
    // 数据指针
    const T* x_ptr = x->data<T>();
    const T* y_ptr = y->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());
    // 数据个数
    int numel = x->numel();
    // out = x + y
    int i = 0;
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] + y_ptr[i];
    }
    VLOG(3) << "AddEqualDim_kernel_end\n";
  }
};

// CPU GradKernel : dx = dout、 dy = dout
template <typename DeviceContext, typename T>
class AddEqualDimGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "AddEqualDimGrad_kernel_start\n";
    // 输入
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    // 输出
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<Tensor>(framework::GradVarName("Y"));

    T* dx_ptr = dx->mutable_data<T>(context.GetPlace());
    T* dy_ptr = dy->mutable_data<T>(context.GetPlace());
    const T* dout_ptr = dout->data<T>();

    int numel = dout->numel();
    int i = 0;
    for (i = 0; i < numel; i++) {
      dx_ptr[i] = dout_ptr[i];
      dy_ptr[i] = dout_ptr[i];
    }
    VLOG(3) << "AddEqualDimGrad_kernel_end\n";
  }
};

}  // namespace operators
}  // namespace paddle
