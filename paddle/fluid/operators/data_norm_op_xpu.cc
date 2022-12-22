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
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/collective_helper.h"
#endif

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

template <typename DeviceContext, typename T>
class  DataNormGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scales = ctx.Input<Tensor>("Scales");
    const auto *means = ctx.Input<Tensor>("Means");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float dr = ctx.Attr<float>("summary_decay_rate");
    const bool need_sync_stats = ctx.Attr<bool>("sync_stats");

    const auto &x_dims = x->dims();
    // Align with CPU version, but should we add this restriction?
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::PreconditionNotMet(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C = x_dims[1];
    // init output
    Tensor *d_x = nullptr;
    T* dx = nullptr;
    if (ctx.HasOutput(framework::GradVarName("X"))) {
      d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
      dx = d_x->mutable_data<T>(ctx.GetPlace());
    }
    T *d_batch_size = ctx.Output<Tensor>(framework::GradVarName("BatchSize"))
                          ->mutable_data<T>(ctx.GetPlace());
    T *d_batch_sum = ctx.Output<Tensor>(framework::GradVarName("BatchSum"))
                         ->mutable_data<T>(ctx.GetPlace());
    T *d_batch_square_sum =
        ctx.Output<Tensor>(framework::GradVarName("BatchSquareSum"))->mutable_data<T>(ctx.GetPlace());
    /*
     * data_norm_grad(Context *ctx, const T* dy, const T* scale, T* x, T* means,
                 T* batch_size, T* batch_square_sum, T* batch_sum, T* dx,
                 T squared_sum_epsilon, int N, int C);
     * */
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::data_norm_grad<T>(dev_ctx.x_context(), d_y->data<T>(),  scales->data<T>(), const_cast<T*>(x->data<T>()), const_cast<T*>(means->data<T>()),
		    d_batch_size, d_batch_square_sum, d_batch_sum, dx,
		    epsilon, N, C);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU data_norm_grad return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
      if (need_sync_stats) {
          //d_batch_size, d_batch_square_sum,d_batch_sum
#if defined(PADDLE_WITH_XPU_BKCL)
	  auto place = ctx.GetPlace();
          auto comm = platform::BKCLCommContext::Instance().Get(0, place);
	  auto stream = dev_ctx.x_context()->xpu_stream;
          PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(), d_batch_size, d_batch_size,
		          C,  BKCL_FLOAT, BKCL_ADD, stream),
			  BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                        "BKCL all reduce failed"));
          PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(), d_batch_square_sum, d_batch_square_sum,
                           C,  BKCL_FLOAT, BKCL_ADD, stream),
			   BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                        "BKCL all reduce failed"));
          PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(), d_batch_sum, d_batch_sum,
                           C,  BKCL_FLOAT, BKCL_ADD, stream),
			   BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                        "BKCL all reduce failed"));
#else
	  PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU, and need_sync_stats connot be "
          "supported on windows now."));
#endif	
      }

      T *batch_size_data =
        ctx.Output<Tensor>("BatchSize")->mutable_data<T>(ctx.GetPlace());

      T *batch_sum_data =
        ctx.Output<Tensor>("BatchSum")->mutable_data<T>(ctx.GetPlace());
      T *batch_square_sum_data =
        ctx.Output<Tensor>("BatchSquareSum")->mutable_data<T>(ctx.GetPlace());
      r = xpu::kernel_update_param<T>(dev_ctx.x_context(), d_batch_size, d_batch_sum, d_batch_square_sum,
                 batch_size_data, batch_sum_data, batch_square_sum_data,  dr,  C);
      PADDLE_ENFORCE_EQ(
         r, XPU_SUCCESS,
        platform::errors::External("XPU kernel_update_param return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};




}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
     data_norm, ops::DataNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
     data_norm_grad, ops::DataNormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
