// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include "paddle/fluid/operators/reduce_ops/reduce_op_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ReduceMaxXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    XPUReduce<DeviceContext, T>(context, xpu::reduce_max<T>);
  }
};

template <typename DeviceContext, typename T>
class ReduceMaxGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto dims = context.Attr<std::vector<int>>("dim");
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Input<Tensor>("Out");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    int in_dtype = context.Attr<int>("in_dtype");
    PADDLE_ENFORCE_EQ(
        in_dtype == -1, true,
        platform::errors::InvalidArgument(
            "XPU only support in_dtype == -1 in reduce_sum_grad op."));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    x_grad->mutable_data<T>(context.GetPlace());
    const T* x_data = x->data<T>();
    const T* out_data = out->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    auto* x_grad_data = x_grad->data<T>();
    const auto& input_dim_size = x->dims().size();
    std::vector<int> true_dims;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        true_dims.push_back(dims[i] + input_dim_size);
      } else {
        true_dims.push_back(dims[i]);
      }
    }
    std::vector<int> ydims(input_dim_size);
    std::vector<int> xdims((input_dim_size));
    std::set<int> dims_set(true_dims.begin(), true_dims.end());
    for (auto i = 0; i < input_dim_size; i++) {
      xdims[i] = x->dims()[i];
      if (dims_set.find(i) != dims_set.end() || reduce_all) {
        ydims[i] = 1;
      } else {
        ydims[i] = x->dims()[i];
      }
    }

    T* brocast1 = nullptr;
    T* brocast2 = nullptr;
    bool* equal = nullptr;
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&brocast1), x->numel() * sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted("XPU has no enough memory"));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&equal), x->numel() * sizeof(bool)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted("XPU has no enough memory"));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&brocast2), x->numel() * sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted("XPU has no enough memory"));

    // step 1. brocast out and out_grad
    int r = xpu::broadcast<T>(dev_ctx.x_context(), out_data, brocast1, ydims,
                              xdims);
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS, true,
        platform::errors::External("XPU broadcast in reduce_max_grad op return"
                                   " wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));
    r = xpu::broadcast<T>(dev_ctx.x_context(), out_grad_data, brocast2, ydims,
                          xdims);
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS, true,
        platform::errors::External("XPU broadcast in reduce_max_grad op return"
                                   " wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));
    // step 2. comparse out_brocast and x
    r = xpu::elementwise_equal<T>(dev_ctx.x_context(), x_data, brocast1, equal,
                                  x->numel());
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS, true,
        platform::errors::External("XPU elementwise_equal in reduce_max_grad "
                                   "op return wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));
    // step 3. get x_grad
    r = xpu::constant<T>(dev_ctx.x_context(), brocast1, x->numel(), 0);
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS, true,
        platform::errors::External("XPU constant in reduce_max_grad op return"
                                   " wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));
    r = xpu::select<T>(dev_ctx.x_context(), equal, brocast2, brocast1,
                       x_grad_data, xdims, xdims);
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS, true,
        platform::errors::External("XPU select in reduce_max_grad op return"
                                   " wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));

    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
    xpu_free(brocast1);
    xpu_free(brocast2);
    xpu_free(equal);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(
    reduce_max,
    ops::ReduceMaxXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    reduce_max_grad,
    ops::ReduceMaxGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
