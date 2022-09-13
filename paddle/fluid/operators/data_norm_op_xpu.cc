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

#include "paddle/fluid/operators/squared_l2_norm_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/framework/data_layout.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class  DataNormXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const std::string& data_layout_str = context.Attr<std::string>("data_layout"); 
    const auto data_layout = framework::StringToDataLayout(data_layout_str);
    const auto *x = context.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    float* scale_w = nullptr;
    float* bias = nullptr;
    bool enable_scale_and_shift = context.Attr<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
        scale_w = const_cast<float*>(context.Input<Tensor>("scale_w")->data<T>());
        bias = const_cast<float*>(context.Input<Tensor>("bias")->data<T>());
    } 
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::InvalidArgument(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C = 
      (data_layout == phi::DataLayout::kNCHW ? x_dims[1] 
                                        : x_dims[x_dims.size() - 1]);
    auto *y = context.Output<Tensor>("Y");
    auto *mean_out = context.Output<Tensor>("Means");
    auto *scales = context.Output<Tensor>("Scales");
    T *y_data = y->mutable_data<T>(context.GetPlace());
    const auto* batch_size = context.Input<Tensor>("BatchSize")->data<T>();
    const auto* batch_sum = context.Input<Tensor>("BatchSum")->data<T>();
    const auto* batch_square_sum = context.Input<Tensor>("BatchSquareSum")->data<T>();
    mean_out->mutable_data<T>(context.GetPlace());
    scales->mutable_data<T>(context.GetPlace());
    //means_arr = b_sum_arr / b_size_arr;  
    T *mean_data = mean_out->data<T>(); 
    T *scale_data = scales->data<T>();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    //  data_norm(Context *context, const float *x, const float *batch_size,
    //                   const float *batch_sum, const float *batch_square_sum, const float *scale_w,
    //                   const float *bias, float *y,  mean_data, scale_data
    //                   int n, int c)
    // N 4096 C 765 slot_dim -1 NCHW  enable false
    int r = xpu::data_norm<T>(dev_ctx.x_context(), x->data<T>(), batch_size, batch_sum, batch_square_sum,
	                      scale_w, bias, y_data, mean_data, scale_data, N, C);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU split kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
     data_norm, ops::DataNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
