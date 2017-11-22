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

#include "paddle/platform/cuda_helper.h"
#include "paddle/operators/roi_pool_op.h"

namespace paddle {
namespace operators {

#define FLT_MAX __FLT_MAX__

constexpr int PADDLE_OPERATORS_ROIPOOL_CUDA_NUM_THREADS = 512;
constexpr int PADDLE_OPERATORS_ROIPOOL_MAXIMUM_NUM_BLOCKS = 4096;

inline int PADDLE_OPERATORS_ROIPOOL_GET_BLOCKS(const int N) {
  return std::min((N + PADDLE_OPERATORS_ROIPOOL_CUDA_NUM_THREADS - 1)
                  / PADDLE_OPERATORS_ROIPOOL_CUDA_NUM_THREADS,
                  PADDLE_OPERATORS_ROIPOOL_MAXIMUM_NUM_BLOCKS);
}

template <typename T>
__global__ void GPURoiPoolForward(
    const int nthreads,
    const T* input_data,
    const int64_t* input_rois,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* output_data,
    int64_t* argmax_data) {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int offset = blockDim.x * gridDim.x;
      for (size_t i = index; i < nthreads; i += offset) {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const int64_t* offset_input_rois = input_rois + n * 5;
        int roi_batch_ind = offset_input_rois[0];
        int roi_start_w = round(offset_input_rois[1] * spatial_scale);
        int roi_start_h = round(offset_input_rois[2] * spatial_scale);
        int roi_end_w = round(offset_input_rois[3] * spatial_scale);
        int roi_end_h = round(offset_input_rois[4] * spatial_scale);

        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        T bin_size_h = static_cast<T>(roi_height)
                      / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width)
                      / static_cast<T>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        T maxval = is_empty ? 0 : -FLT_MAX;
        int maxidx = -1;
        const T* offset_input_data =
            input_data + (roi_batch_ind * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int input_data_index = h * width + w;
            if (offset_input_data[input_data_index] > maxval) {
              maxval = offset_input_data[input_data_index];
              maxidx = input_data_index;
            }
          }
        }
        output_data[index] = maxval;
        if (argmax_data) {
          argmax_data[index] = maxidx;
        }
    }
  }

template <typename T>
__global__ void GPURoiPoolBackward(
    const int nthreads,
    const int64_t* input_rois,
    const T* output_grad,
    const int64_t* argmax_data,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* input_grad) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = index; i < nthreads; i += offset) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int64_t* offset_input_rois = input_rois + n * 5;
      int roi_batch_ind = offset_input_rois[0];
      int input_offset = (roi_batch_ind * channels + c) * height * width;
      int output_offset = (n * channels + c) * pooled_height * pooled_width;
      const T* offset_output_grad = output_grad + output_offset;
      T* offset_input_grad = input_grad + input_offset;
      const int64_t* offset_argmax_data = argmax_data + output_offset;

      int argmax = offset_argmax_data[ph * pooled_width + pw];
      if (argmax != -1) {
        platform::CudaAtomicAdd(offset_input_grad + argmax,
          static_cast<T>(offset_output_grad[ph * pooled_width + pw]));
      }
    }
  }


template <typename Place, typename T>
class GPURoiPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<Tensor>("Rois");
    auto* out = ctx.Output<Tensor>("Out");
    auto* argmax = ctx.Output<Tensor>("Argmax");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    PADDLE_ENFORCE_GT(pooled_height, 0,
                      "The pooled output height must greater than 0");
    PADDLE_ENFORCE_GT(pooled_width, 0,
                      "The pooled output width must greater than 0");
    PADDLE_ENFORCE_GT(spatial_scale, 0,
                      "The spatial scale must greater than 0");

    auto in_dims = in->dims();
    auto in_stride = framework::stride(in_dims);
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];
    auto out_dims = in_dims;
    out_dims[0] = rois_num;
    out_dims[1] = in_dims[1];
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;

    out->Resize(out_dims);
    out->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<Place, T> set_zero;
    set_zero(ctx.device_context(), out, static_cast<T>(0));
    argmax->Resize(out->dims());
    argmax->mutable_data<int64_t>(ctx.GetPlace());
    math::SetConstant<Place, int64_t> set_init;
    set_init(ctx.device_context(), argmax, static_cast<int64_t>(-1));

    if (rois_num== 0) return;

    int output_size = out->numel();
    int blocks = PADDLE_OPERATORS_ROIPOOL_GET_BLOCKS(output_size);
    int threads = PADDLE_OPERATORS_ROIPOOL_CUDA_NUM_THREADS;

    GPURoiPoolForward<T>
      <<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
      output_size,
      in->data<T>(),
      rois->data<int64_t>(),
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      out->mutable_data<T>(ctx.GetPlace()),
      argmax->mutable_data<int64_t>(ctx.GetPlace()));

      return;
  }
};

template <typename Place, typename T>
class GPURoiPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<Tensor>("Rois");
    auto* argmax = ctx.Input<Tensor>("Argmax");

    auto* out_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (x_grad) {
      x_grad->Resize(in->dims());
      x_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<Place, T> set_zero;
      set_zero(ctx.device_context(), x_grad, static_cast<T>(0));

      int output_grad_size = out_grad->numel();
      int blocks = PADDLE_OPERATORS_ROIPOOL_GET_BLOCKS(output_grad_size);
      int threads = PADDLE_OPERATORS_ROIPOOL_CUDA_NUM_THREADS;

      if (output_grad_size > 0) {
        GPURoiPoolBackward<T>
          <<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
          output_grad_size,
          rois->data<int64_t>(),
          out_grad->data<T>(),
          argmax->data<int64_t>(),
          rois_num,
          spatial_scale,
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          x_grad->mutable_data<T>(ctx.GetPlace()));
        }
      return;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    roi_pool,
    ops::GPURoiPoolOpKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    roi_pool_grad,
    ops::GPURoiPoolGradOpKernel<paddle::platform::GPUPlace, float>);
