/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <string>
#include "paddle/fluid/operators/interpolate_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeNearestNeighborInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);

    int out_img_idx = tid % out_img_w;
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    out[tid] = in[out_id_h * input_w + channel_id * in_img_size +
                  in_img_idy * in_img_w + in_img_idx];
  }
}

template <typename T>
__global__ void KeNearestNeighborInterpBw(
    T* in, const size_t in_img_h, const size_t in_img_w, const size_t input_h,
    const size_t input_w, const T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);

    int out_img_idx = tid % out_img_w;
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idy * in_img_w + in_img_idx];
    const T out_pos = out[out_id_h * output_w + out_id_w];
    platform::CudaAtomicAdd(in_pos, out_pos);
  }
}

template <typename T>
__global__ void KeBilinearInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const bool align_corners, const int align_mode) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T h1lambda = align_flag ? ratio_h * (out_img_idy + 0.5) - 0.5 - in_img_idy
                            : ratio_h * out_img_idy - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    int out_img_idx = tid % out_img_w;
    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T w1lambda = align_flag ? ratio_w * (out_img_idx + 0.5) - 0.5 - in_img_idx
                            : ratio_w * out_img_idx - in_img_idx;
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
    const size_t num_channels, const T ratio_h, const T ratio_w,
    const bool align_corners, const int align_mode) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = align_flag ? ratio_h * (out_img_idy + 0.5) - 0.5
                                : ratio_h * out_img_idy;
    in_img_idy = (in_img_idy > 0) ? in_img_idy : 0;
    int h_id = (in_img_idy < in_img_h - 1) ? 1 : 0;
    T h1lambda = align_flag ? ratio_h * (out_img_idy + 0.5) - 0.5 - in_img_idy
                            : ratio_h * out_img_idy - in_img_idy;

    T h2lambda = 1.f - h1lambda;

    int out_img_idx = tid % out_img_w;
    int in_img_idx = align_flag ? ratio_w * (out_img_idx + 0.5) - 0.5
                                : ratio_w * out_img_idx;
    in_img_idx = (in_img_idx > 0) ? in_img_idx : 0;
    int w_id = (in_img_idx < in_img_w - 1) ? 1 : 0;
    T w1lambda = align_flag ? ratio_w * (out_img_idx + 0.5) - 0.5 - in_img_idx
                            : ratio_w * out_img_idx - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idy * in_img_w + in_img_idx];
    const T* out_pos = &out[out_id_h * output_w + out_id_w];
    platform::CudaAtomicAdd(&in_pos[0], h2lambda * w2lambda * out_pos[0]);
    platform::CudaAtomicAdd(&in_pos[w_id], h2lambda * w1lambda * out_pos[0]);
    platform::CudaAtomicAdd(&in_pos[h_id * in_img_w],
                            h1lambda * w2lambda * out_pos[0]);
    platform::CudaAtomicAdd(&in_pos[h_id * in_img_w + w_id],
                            h1lambda * w1lambda * out_pos[0]);
  }
}

template <typename T>
class InterpolateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* input_data = input->data<T>();

    int n = input->dims()[0];
    int c = input->dims()[1];
    int in_h = input->dims()[2];
    int in_w = input->dims()[3];

    auto interp_method = ctx.Attr<std::string>("interp_method");
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");

    float scale = ctx.Attr<float>("scale");
    if (scale > 0) {
      out_h = in_h * scale;
      out_w = in_w * scale;
    }

    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      Tensor sizes;
      framework::TensorCopy(*out_size, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }

    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    auto* output_data =
        output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = c * in_hw;
    int out_chw = c * out_hw;

    float ratio_h = 0.f;
    float ratio_w = 0.f;
    if (out_h > 1) {
      ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                                : static_cast<float>(in_h) / out_h;
    }
    if (out_w > 1) {
      ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                                : static_cast<float>(in_w) / out_w;
    }

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    int pixelNum = n * out_chw;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    if ("nearest" == interp_method) {
      KeNearestNeighborInterpFw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
          out_chw, c, ratio_h, ratio_w, align_corners);
    } else if ("bilinear" == interp_method) {
      KeBilinearInterpFw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_data, in_h, in_w, n, in_chw, output_data, out_h, out_w, n,
          out_chw, c, ratio_h, ratio_w, align_corners, align_mode);
    }
  }
};

template <typename T>
class InterpolateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* output_grad_data = output_grad->data<T>();
    auto* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

    auto& device_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> zero;
    zero(device_ctx, input_grad, static_cast<T>(0.0));

    int n = input_grad->dims()[0];
    int c = input_grad->dims()[1];
    int in_h = input_grad->dims()[2];
    int in_w = input_grad->dims()[3];

    auto interp_method = ctx.Attr<std::string>("interp_method");
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale = ctx.Attr<float>("scale");
    if (scale > 0) {
      out_h = in_h * scale;
      out_w = in_w * scale;
    }
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      Tensor sizes;
      framework::TensorCopy(*out_size, platform::CPUPlace(), &sizes);
      auto size_data = sizes.data<int>();
      out_h = size_data[0];
      out_w = size_data[1];
    }

    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = c * in_hw;
    int out_chw = c * out_hw;

    float ratio_h = 0.f;
    float ratio_w = 0.f;
    if (out_h > 1) {
      ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                                : static_cast<float>(in_h) / out_h;
    }
    if (out_w > 1) {
      ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                                : static_cast<float>(in_w) / out_w;
    }

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    int pixelNum = n * out_chw;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    if ("nearest" == interp_method) {
      KeNearestNeighborInterpBw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h,
          out_w, n, out_chw, c, ratio_h, ratio_w, align_corners);
    } else if ("bilinear" == interp_method) {
      KeBilinearInterpBw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_grad_data, in_h, in_w, n, in_chw, output_grad_data, out_h,
          out_w, n, out_chw, c, ratio_h, ratio_w, align_corners, align_mode);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bilinear_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bilinear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(nearest_interp, ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(nearest_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
