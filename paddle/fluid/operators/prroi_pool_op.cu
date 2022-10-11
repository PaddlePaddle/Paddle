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

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void GPUPRROIPoolForward(const int nthreads,
                                    const T* input_data,
                                    const T* input_rois,
                                    const float spatial_scale,
                                    const int input_channels,
                                    const int height,
                                    const int width,
                                    const int output_channels,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int* rois_batch_id_data,
                                    T* output_data) {
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
    int input_channel = c;
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
              offset_input_data,
              h_iter,
              w_iter,
              h_iter + 1,
              w_iter + 1,
              max(win_start_h, static_cast<T>(h_iter)),
              max(win_start_w, static_cast<T>(w_iter)),
              min(win_end_h, static_cast<T>(h_iter) + static_cast<T>(1.0)),
              min(win_end_w, static_cast<T>(w_iter) + static_cast<T>(1.0)),
              height,
              width);
        }
      }
      output_data[i] = sum_out / win_size;
    } else {
      output_data[i] = 0.;
    }
  }
}

template <typename T>
__global__ void GPUPRROIPoolBackward(const int nthreads,
                                     const T* in_data,
                                     const T* input_rois,
                                     const T* output_grad_data,
                                     const float spatial_scale,
                                     const int input_channels,
                                     const int height,
                                     const int width,
                                     const int output_channels,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int* rois_batch_id_data,
                                     T* input_grad_data,
                                     const T* out_data,
                                     T* input_roi_grad_data) {
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
    int input_channel = c;
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
    T* offset_input_roi_grad_data = input_roi_grad_data + n * 4;

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
        PrRoIPoolingMatDistributeDiff<T>(
            offset_input_grad_data,
            sum_out,
            h_iter,
            w_iter,
            h_iter + 1,
            w_iter + 1,
            max(win_start_h, static_cast<T>(h_iter)),
            max(win_start_w, static_cast<T>(w_iter)),
            min(win_end_h, static_cast<T>(h_iter) + static_cast<T>(1.0)),
            min(win_end_w, static_cast<T>(w_iter) + static_cast<T>(1.0)),
            height,
            width);
      }
    }

    const T* offset_out_data = out_data + i;
    const T* offset_in_data = in_data + input_offset;
    PrRoIPoolingCoorBackward<T>(s_w,
                                e_w,
                                s_h,
                                e_h,
                                width,
                                height,
                                win_start_w,
                                win_start_h,
                                win_end_w,
                                win_end_h,
                                pw,
                                ph,
                                pooled_width,
                                pooled_height,
                                win_size,
                                spatial_scale,
                                offset_in_data,
                                offset_out_data,
                                offset_input_roi_grad_data,
                                offset_output_grad_data);
  }
}

template <typename T>
class GPUPRROIPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<phi::DenseTensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    auto output_channels = input_channels;
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];
    if (rois_num == 0) return;

    // set rois batch id
    phi::DenseTensor rois_batch_id_list;
    rois_batch_id_list.Resize({rois_num});
    int* rois_batch_id_data =
        rois_batch_id_list.mutable_data<int>(platform::CPUPlace());

    if (ctx.HasInput("BatchRoINums") || rois->lod().empty()) {
      auto* batchroinum = ctx.Input<phi::DenseTensor>("BatchRoINums");
      phi::DenseTensor batch_index_cpu;
      framework::TensorCopySync(
          *batchroinum, platform::CPUPlace(), &batch_index_cpu);

      int rois_batch_size = batchroinum->dims()[0];
      auto* batch_index = batch_index_cpu.data<int64_t>();
      size_t c = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int64_t k = 0; k < batch_index[n]; ++k) {
          rois_batch_id_data[c] = n;
          c = c + 1;
        }
      }

    } else {
      auto rois_lod = rois->lod().back();
      int rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          rois_batch_size,
          batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and input(X) batch_size must be the same."));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num,
          rois_num_with_lod,
          platform::errors::InvalidArgument(
              "The rois_num from input and lod must be the same."));

      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
          rois_batch_id_data[i] = n;
        }
      }
    }

    int output_size = out->numel();
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    auto cplace = platform::CPUPlace();
    auto& dev_ctx = ctx.cuda_device_context();
    int bytes = rois_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = memory::Alloc(
        dev_ctx.GetPlace(),
        bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    const auto gplace = ctx.GetPlace();
    memory::Copy(gplace,
                 roi_id_data,
                 cplace,
                 rois_batch_id_data,
                 bytes,
                 dev_ctx.stream());

    // call cuda kernel function
    GPUPRROIPoolForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size,
        in->data<T>(),
        rois->data<T>(),
        spatial_scale,
        input_channels,
        height,
        width,
        output_channels,
        pooled_height,
        pooled_width,
        roi_id_data,
        out->mutable_data<T>(ctx.GetPlace()));
  }
};

template <typename DeviceContext, typename T>
class GPUPRROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<phi::DenseTensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Input<phi::DenseTensor>("Out");

    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* input_roi_grad =
        ctx.Output<LoDTensor>(framework::GradVarName("ROIs"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int input_channels = in->dims()[1];
    auto output_channels = input_channels;
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (input_grad || input_roi_grad) {
      // set roi batch id
      phi::DenseTensor rois_batch_id_list;
      rois_batch_id_list.Resize({rois_num});
      int* rois_batch_id_data =
          rois_batch_id_list.mutable_data<int>(platform::CPUPlace());

      if (ctx.HasInput("BatchRoINums") || rois->lod().empty()) {
        auto* batchroinum = ctx.Input<phi::DenseTensor>("BatchRoINums");
        phi::DenseTensor batch_index_cpu;
        framework::TensorCopySync(
            *batchroinum, platform::CPUPlace(), &batch_index_cpu);

        int rois_batch_size = batchroinum->dims()[0];
        auto* batch_index = batch_index_cpu.data<int64_t>();
        size_t c = 0;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (int64_t k = 0; k < batch_index[n]; ++k) {
            rois_batch_id_data[c] = n;
            c = c + 1;
          }
        }
      } else {
        PADDLE_ENFORCE_EQ(rois->lod().empty(),
                          false,
                          platform::errors::InvalidArgument(
                              "the lod of Input ROIs should not be empty when "
                              "BatchRoINums is None!"));
        auto rois_lod = rois->lod().back();
        int rois_batch_size = rois_lod.size() - 1;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            rois_batch_id_data[i] = n;
          }
        }
      }

      auto cplace = platform::CPUPlace();
      auto& dev_ctx = ctx.cuda_device_context();
      int bytes = rois_batch_id_list.numel() * sizeof(int);
      auto roi_ptr = memory::Alloc(
          dev_ctx.GetPlace(),
          bytes,
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
      const auto gplace = ctx.GetPlace();
      memory::Copy(gplace,
                   roi_id_data,
                   cplace,
                   rois_batch_id_data,
                   bytes,
                   dev_ctx.stream());

      input_grad->mutable_data<T>(ctx.GetPlace());
      phi::funcs::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.cuda_device_context(), input_grad, static_cast<T>(0));
      input_roi_grad->mutable_data<T>(ctx.GetPlace());
      set_zero(ctx.cuda_device_context(), input_roi_grad, static_cast<T>(0));

      int output_grad_size = output_grad->numel();
      int blocks = NumBlocks(output_grad_size);
      int threads = kNumCUDAThreads;

      if (output_grad_size > 0) {
        GPUPRROIPoolBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            output_grad_size,
            in->data<T>(),
            rois->data<T>(),
            output_grad->data<T>(),
            spatial_scale,
            input_channels,
            height,
            width,
            output_channels,
            pooled_height,
            pooled_width,
            roi_id_data,
            input_grad->mutable_data<T>(ctx.GetPlace()),
            out->data<T>(),
            input_roi_grad->mutable_data<T>(ctx.GetPlace()));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(prroi_pool,
                        ops::GPUPRROIPoolOpKernel<float>,
                        ops::GPUPRROIPoolOpKernel<double>);
REGISTER_OP_CUDA_KERNEL(prroi_pool_grad,
                        ops::GPUPRROIPoolGradOpKernel<phi::GPUContext, float>,
                        ops::GPUPRROIPoolGradOpKernel<phi::GPUContext, double>);
