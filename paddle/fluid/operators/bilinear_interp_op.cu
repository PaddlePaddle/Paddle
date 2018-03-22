/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/bilinear_interp_op.cu.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class BilinearInterpOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input_t = ctx.Input<Tensor>("X");      // float tensor
    auto* output_t = ctx.Output<Tensor>("Out");  // float tensor
    auto* input = input_t->data<T>();
    auto* output = output_t->mutable_data<T>(ctx.GetPlace());

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    int batch_size = input_t->dims()[0];
    int channels = input_t->dims()[1];
    int in_h = input_t->dims()[2];
    int in_w = input_t->dims()[3];

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = channels * in_hw;
    int out_chw = channels * out_hw;

    T ratio_h = (out_h > 1) ? static_cast<T>(in_h - 1) / (out_h - 1) : 0.f;
    T ratio_w = (out_w > 1) ? static_cast<T>(in_w - 1) / (out_w - 1) : 0.f;

    if (in_h == out_h && in_w == out_w) {
      memcpy(output, input, input_t->numel() * sizeof(T));
    } else {
      int threadNum = batch_size * out_chw;
      int blocks = (threadNum + 1024 - 1) / 1024;

      KeBilinearInterpFw<
          T><<<blocks, 1024, 0, ctx.cuda_device_context().stream()>>>(
          input, in_h, in_w, batch_size, in_chw, output, out_h, out_w,
          batch_size, out_chw, channels, ratio_h, ratio_w);
    }
  }
};

template <typename T>
class BilinearInterpGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_input_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_output_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_input = d_input_t->mutable_data<T>(ctx.GetPlace());
    auto* d_output = d_output_t->data<T>();

    auto& device_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> zero;
    zero(device_ctx, d_input_t, static_cast<T>(0.0));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    int batch_size = d_input_t->dims()[0];
    int channels = d_input_t->dims()[1];
    int in_h = d_input_t->dims()[2];
    int in_w = d_input_t->dims()[3];

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = channels * in_hw;
    int out_chw = channels * out_hw;

    T ratio_h = (out_h > 1) ? static_cast<T>(in_h - 1) / (out_h - 1) : 0.f;
    T ratio_w = (out_w > 1) ? static_cast<T>(in_w - 1) / (out_w - 1) : 0.f;

    if (in_h == out_h && in_w == out_w) {
      memcpy(d_input, d_output, d_input_t->numel() * sizeof(T));
    } else {
      int threadNum = batch_size * out_chw;
      int blocks = (threadNum + 1024 - 1) / 1024;

      KeBilinearInterpBw<
          T><<<blocks, 1024, 0, ctx.cuda_device_context().stream()>>>(
          d_input, in_h, in_w, batch_size, in_chw, d_output, out_h, out_w,
          batch_size, out_chw, channels, ratio_h, ratio_w);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bilinear_interp,
                        ops::BilinearInterpOpCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(bilinear_interp_grad,
                        ops::BilinearInterpGradOpCUDAKernel<float>);
