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
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/roi_pool_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void GPUROIPoolBackward(
    const int nthreads, const T* input_rois, const T* output_grad,
    const int64_t* argmax_data, const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, int* roi_batch_id_data,
    T* input_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    int roi_batch_ind = roi_batch_id_data[n];
    int input_offset = (roi_batch_ind * channels + c) * height * width;
    int output_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_output_grad = output_grad + output_offset;
    T* offset_input_grad = input_grad + input_offset;
    const int64_t* offset_argmax_data = argmax_data + output_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      platform::CudaAtomicAdd(
          offset_input_grad + argmax,
          static_cast<T>(offset_output_grad[ph * pooled_width + pw]));
    }
  }
}

template <typename Place, typename T>
class GPUROIPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* rois_lod = ctx.Input<Tensor>("RoisNum");
    auto* argmax = ctx.Input<Tensor>("Argmax");

    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int channels = in->dims()[1];
    int height = in->dims()[2];
    int width = in->dims()[3];

    if (x_grad) {
      framework::Tensor roi_batch_id_list;
      roi_batch_id_list.Resize({rois_num});
      auto cplace = platform::CPUPlace();
      int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);

      auto& dev_ctx = ctx.cuda_device_context();
      auto gplace = ctx.GetPlace();
      if (ctx.HasInput("RoisNum")) {
        auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
        int rois_batch_size = rois_num_t->numel();
        std::vector<int> rois_num_list(rois_batch_size);
        memory::Copy(cplace, rois_num_list.data(), gplace,
                     rois_num_t->data<int>(), sizeof(int) * rois_batch_size, 0);
        int start = 0;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (int i = start; i < start + rois_num_list[n]; ++i) {
            roi_batch_id_data[i] = n;
          }
          start += rois_num_list[n];
        }
      } else {
        auto rois_lod = rois->lod().back();
        int rois_batch_size = rois_lod.size() - 1;
        for (int n = 0; n < rois_batch_size; ++n) {
          for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
            roi_batch_id_data[i] = n;
          }
        }
      }
      int bytes = roi_batch_id_list.numel() * sizeof(int);
      auto roi_ptr = memory::Alloc(dev_ctx, bytes);
      int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
      memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                   dev_ctx.stream());

      x_grad->mutable_data<T>(ctx.GetPlace());
      phi::funcs::SetConstant<Place, T> set_zero;
      set_zero(dev_ctx, x_grad, static_cast<T>(0));

      int output_grad_size = out_grad->numel();
      int blocks = NumBlocks(output_grad_size);
      int threads = kNumCUDAThreads;

      if (output_grad_size > 0) {
        GPUROIPoolBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
            output_grad_size, rois->data<T>(), out_grad->data<T>(),
            argmax->data<int64_t>(), rois_num, spatial_scale, channels, height,
            width, pooled_height, pooled_width, roi_id_data,
            x_grad->mutable_data<T>(ctx.GetPlace()));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roi_pool_grad,
    ops::GPUROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIPoolGradOpKernel<paddle::platform::CUDADeviceContext, double>);
