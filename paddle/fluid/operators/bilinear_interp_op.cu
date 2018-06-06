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

#include "paddle/fluid/operators/bilinear_interp_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeBilinearInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const T ratio_h, const T ratioW) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nthreads) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = ratio_h * out_img_idy;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T h1lambda = ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int out_img_idx = tid % out_img_w;
    int in_img_idx = ratioW * out_img_idx;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T w1lambda = ratioW * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    const T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                          in_img_idy * in_img_w + in_img_idx];

    // bilinear interpolation
    out[out_id_h * output_w + out_id_w] =
        h2lambda * (w2lambda * in_pos[0] + w1lambda * in_pos[w_id]) +
        h1lambda * (w2lambda * in_pos[h_id * in_img_w] +
                    w1lambda * in_pos[h_id * in_img_w + w_id]);
  }
}

template <typename T>
__global__ void KeBilinearInterpBw(
    T* in, const size_t in_img_h, const size_t in_img_w, const size_t input_h,
    const size_t input_w, const T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const T ratio_h, const T ratioW) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nthreads) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = ratio_h * out_img_idy;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T h1lambda = ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int out_img_idx = tid % out_img_w;
    int in_img_idx = ratioW * out_img_idx;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T w1lambda = ratioW * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idy * in_img_w + in_img_idx];
    const T* out_pos = &out[out_id_h * output_w + out_id_w];
    atomicAdd(&in_pos[0], h2lambda * w2lambda * out_pos[0]);
    atomicAdd(&in_pos[w_id], h2lambda * w1lambda * out_pos[0]);
    atomicAdd(&in_pos[h_id * in_img_w], h1lambda * w2lambda * out_pos[0]);
    atomicAdd(&in_pos[h_id * in_img_w + w_id],
              h1lambda * w1lambda * out_pos[0]);
  }
}

template <typename T>
class BilinearInterpOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input_t = ctx.Input<Tensor>("X");      // float tensor
    auto* output_t = ctx.Output<Tensor>("Out");  // float tensor
    auto* input = input_t->data<T>();

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    auto out_dims = output_t->dims();
    auto out_size_t = ctx.Input<Tensor>("OutSize");
    if (out_size_t != nullptr) {
      Tensor sizes;
      framework::TensorCopy(*out_size_t, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }
    auto* output = output_t->mutable_data<T>(
        {out_dims[0], out_dims[1], out_h, out_w}, ctx.GetPlace());

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
    auto* d_output = d_output_t->data<T>();
    auto* d_input = d_input_t->mutable_data<T>(ctx.GetPlace());

    auto& device_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> zero;
    zero(device_ctx, d_input_t, static_cast<T>(0.0));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");

    auto out_size_t = ctx.Input<Tensor>("OutSize");
    if (out_size_t != nullptr) {
      Tensor sizes;
      framework::TensorCopy(*out_size_t, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }

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
