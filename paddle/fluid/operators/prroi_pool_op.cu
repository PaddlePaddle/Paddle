/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/prroi_pool_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
DEVICE void PrRoIPoolingDistributeDiffCUDA(T* diff, const T top_diff,
                                           const int h, const int w,
                                           const int height, const int width,
                                           const T coeff) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  if (!overflow) {
    paddle::platform::CudaAtomicAdd(diff + h * width + w, top_diff * coeff);
  }
}

template <typename T>
__global__ void GPUPRROIPoolForward(
    const int nthreads, const T* input_data, const T* input_rois,
    const float spatial_scale, const int input_channels, const int height,
    const int width, const int output_channels, const int pooled_height,
    const int pooled_width, const int* rois_batch_id_data, T* output_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    // The output is in order (n, c, ph, pw)
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % output_channels;
    int n = i / pooled_width / pooled_height / output_channels;

    // set roi_batch_id
    int roi_batch_id = rois_batch_id_data[n];

    // [start, end) interval for spatial sampling
    const T* offset_input_rois = input_rois + n * 4;
    T roi_start_w = static_cast<T>(offset_input_rois[0]) * spatial_scale;
    T roi_start_h = static_cast<T>(offset_input_rois[1]) * spatial_scale;
    T roi_end_w = static_cast<T>(offset_input_rois[2]) * spatial_scale;
    T roi_end_h = static_cast<T>(offset_input_rois[3]) * spatial_scale;

    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.0));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.0));

    // Compute w and h at input feature map
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T win_start_w = roi_start_w + bin_size_w * pw;
    T win_start_h = roi_start_h + bin_size_h * ph;
    T win_end_w = win_start_w + bin_size_w;
    T win_end_h = win_start_h + bin_size_h;

    T win_size = max(static_cast<T>(0.0), bin_size_w * bin_size_h);
    int input_channel = (c * pooled_height + ph) * pooled_width + pw;
    const T* offset_input_data =
        input_data +
        (roi_batch_id * input_channels + input_channel) * height * width;

    if (win_size > static_cast<T>(0.0)) {
      int s_w = floor(win_start_w);
      int e_w = ceil(win_end_w);
      int s_h = floor(win_start_h);
      int e_h = ceil(win_end_h);
      T sum_out = 0;

      for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
        for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
          sum_out += PrRoIPoolingMatCalculation(
              offset_input_data, h_iter, w_iter, h_iter + 1, w_iter + 1,
              max(win_start_h, static_cast<T>(h_iter)),
              max(win_start_w, static_cast<T>(w_iter)),
              min(win_end_h, static_cast<T>(h_iter) + static_cast<T>(1.0)),
              min(win_end_w, static_cast<T>(w_iter) + static_cast<T>(1.0)),
              height, width);
        }
      }
      output_data[i] = sum_out / win_size;
    } else {
      output_data[i] = 0.;
    }
  }
}

template <typename T>
__global__ void GPUPRROIPoolBackward(
    const int nthreads, const T* input_rois, const T* output_grad_data,
    const float spatial_scale, const int input_channels, const int height,
    const int width, const int output_channels, const int pooled_height,
    const int pooled_width, const int* rois_batch_id_data, T* input_grad_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    // The output is in order (n, c, ph, pw)
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % output_channels;
    int n = i / pooled_width / pooled_height / output_channels;

    // set roi_batch_id
    int roi_batch_id = rois_batch_id_data[n];
    int input_channel = (c * pooled_height + ph) * pooled_width + pw;
    int input_offset =
        (roi_batch_id * input_channels + input_channel) * height * width;
    T* offset_input_grad_data = input_grad_data + input_offset;
    const T* offset_output_grad_data = output_grad_data + i;

    // [start, end) interval for spatial sampling
    const T* offset_input_rois = input_rois + n * 4;
    T roi_start_w = static_cast<T>(offset_input_rois[0]) * spatial_scale;
    T roi_start_h = static_cast<T>(offset_input_rois[1]) * spatial_scale;
    T roi_end_w = static_cast<T>(offset_input_rois[2]) * spatial_scale;
    T roi_end_h = static_cast<T>(offset_input_rois[3]) * spatial_scale;

    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.0));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.0));

    // Compute w and h at input feature map
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T win_start_w = roi_start_w + bin_size_w * pw;
    T win_start_h = roi_start_h + bin_size_h * ph;
    T win_end_w = win_start_w + bin_size_w;
    T win_end_h = win_start_h + bin_size_h;

    T win_size = max(static_cast<T>(0.0), bin_size_w * bin_size_h);
    int s_w = floor(win_start_w);
    int e_w = ceil(win_end_w);
    int s_h = floor(win_start_h);
    int e_h = ceil(win_end_h);

    T sum_out = win_size == static_cast<T>(0.)
                    ? static_cast<T>(0.)
                    : *offset_output_grad_data / win_size;

    for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
      for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
        PrRoIPoolingMatDistributeDiff(
            offset_input_grad_data, sum_out, h_iter, w_iter, h_iter + 1,
            w_iter + 1, max(win_start_h, static_cast<T>(h_iter)),
            max(win_start_w, static_cast<T>(w_iter)),
            min(win_end_h, static_cast<T>(h_iter) + static_cast<T>(1.0)),
            min(win_end_w, static_cast<T>(w_iter) + static_cast<T>(1.0)),
            height, width, PrRoIPoolingDistributeDiffCUDA<T>);
      }
    }
  }
}

template <typename T>
class GPUPRROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto output_channels = ctx.Attr<int>("output_channels");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    PADDLE_ENFORCE_EQ(input_channels,
                      output_channels * pooled_height * pooled_width,
                      "the channels of input X should equal the product of "
                      "output_channels x pooled_height x pooled_width");

    int rois_num = rois->dims()[0];
    if (rois_num == 0) return;

    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size, batch_size,
        "The rois_batch_size and input(X) batch_size must be the same.");
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(rois_num, rois_num_with_lod,
                      "The rois_num from input and lod must be the same.");

    // set rois batch id
    framework::Tensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num});
    int* rois_batch_id_data =
        rois_batch_id_list.mutable_data<int>(platform::CPUPlace());
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        rois_batch_id_data[i] = n;
      }
    }

    framework::Tensor rois_batch_id_list_gpu;
    framework::TensorCopy(rois_batch_id_list, ctx.GetPlace(),
                          ctx.device_context(), &rois_batch_id_list_gpu);

    int output_size = out->numel();
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    // call cuda kernel function
    GPUPRROIPoolForward<
        T><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
        output_size, in->data<T>(), rois->data<T>(), spatial_scale,
        input_channels, height, width, output_channels, pooled_height,
        pooled_width, rois_batch_id_list_gpu.data<int>(),
        out->mutable_data<T>(ctx.GetPlace()));
  }
};

template <typename DeviceContext, typename T>
class GPUPRROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");

    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto output_channels = ctx.Attr<int>("output_channels");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int input_channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (input_grad) {
      // set roi batch id
      framework::Tensor rois_batch_id_list;
      rois_batch_id_list.Resize({rois_num});
      int* rois_batch_id_data =
          rois_batch_id_list.mutable_data<int>(platform::CPUPlace());
      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
      }

      framework::Tensor rois_batch_id_list_gpu;
      framework::TensorCopy(rois_batch_id_list, ctx.GetPlace(),
                            ctx.device_context(), &rois_batch_id_list_gpu);

      input_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.cuda_device_context(), input_grad, static_cast<T>(0));

      int output_grad_size = output_grad->numel();
      int blocks = NumBlocks(output_grad_size);
      int threads = kNumCUDAThreads;

      if (output_grad_size > 0) {
        GPUPRROIPoolBackward<
            T><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
            output_grad_size, rois->data<T>(), output_grad->data<T>(),
            spatial_scale, input_channels, height, width, output_channels,
            pooled_height, pooled_width, rois_batch_id_list_gpu.data<int>(),
            input_grad->mutable_data<T>(ctx.GetPlace()));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(prroi_pool, ops::GPUPRROIPoolOpKernel<float>,
                        ops::GPUPRROIPoolOpKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    prroi_pool_grad,
    ops::GPUPRROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUPRROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, double>);
