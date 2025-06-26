// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/roi_pool_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline uint32_t NumBlocks(const int64_t N) {
  return static_cast<uint32_t>(
      std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
               static_cast<int64_t>(kNumMaximumNumBlocks)));
}

template <typename T, typename IndexType>
__global__ void GPURoiPoolBackward(const IndexType nthreads,
                                   const T* input_rois,
                                   const T* output_grad,
                                   const int64_t* arg_max_data,
                                   const IndexType num_rois,
                                   const float spatial_scale,
                                   const IndexType channels,
                                   const IndexType height,
                                   const IndexType width,
                                   const int pooled_height,
                                   const int pooled_width,
                                   int* box_batch_id_data,
                                   T* input_grad) {
  IndexType index =
      static_cast<IndexType>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexType offset = static_cast<IndexType>(blockDim.x) * gridDim.x;
  for (IndexType i = index; i < nthreads; i += offset) {
    IndexType pw = i % pooled_width;
    IndexType ph = (i / pooled_width) % pooled_height;
    IndexType c = (i / pooled_width / pooled_height) % channels;
    IndexType n = i / pooled_width / pooled_height / channels;

    int roi_batch_ind = box_batch_id_data[n];
    IndexType input_offset = (roi_batch_ind * channels + c) * height * width;
    IndexType output_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_output_grad = output_grad + output_offset;
    T* offset_input_grad = input_grad + input_offset;
    const int64_t* offset_arg_max_data = arg_max_data + output_offset;

    int64_t arg_max = offset_arg_max_data[ph * pooled_width + pw];
    if (arg_max != -1) {
      phi::CudaAtomicAdd(
          offset_input_grad + arg_max,
          static_cast<T>(offset_output_grad[ph * pooled_width + pw]));
    }
  }
}

template <typename T, typename Context>
void RoiPoolGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& boxes,
                       const paddle::optional<DenseTensor>& boxes_num,
                       const DenseTensor& arg_max,
                       const DenseTensor& out_grad,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       DenseTensor* dx) {
  auto x_dims = x.dims();
  int64_t channels = x_dims[1];
  int64_t height = x_dims[2];
  int64_t width = x_dims[3];
  int64_t rois_num = boxes.dims()[0];

  if (dx) {
    DenseTensor box_batch_id_list;
    box_batch_id_list.Resize({rois_num});
    int* box_batch_id_data =
        dev_ctx.template HostAlloc<int>(&box_batch_id_list);

    auto gplace = dev_ctx.GetPlace();
    if (boxes_num) {
      int boxes_batch_size = boxes_num->numel();
      std::vector<int> boxes_num_list(boxes_batch_size);
      memory_utils::Copy(phi::CPUPlace(),
                         boxes_num_list.data(),
                         gplace,
                         boxes_num->data<int>(),
                         sizeof(int) * boxes_batch_size,
                         0);
      int start = 0;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (int i = start; i < start + boxes_num_list[n]; ++i) {
          box_batch_id_data[i] = n;
        }
        start += boxes_num_list[n];
      }
    } else {
      auto boxes_lod = boxes.lod().back();
      int boxes_batch_size = boxes_lod.size() - 1;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
          box_batch_id_data[i] = n;
        }
      }
    }
    int bytes = box_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    memory_utils::Copy(gplace,
                       roi_id_data,
                       phi::CPUPlace(),
                       box_batch_id_data,
                       bytes,
                       dev_ctx.stream());

    dev_ctx.template Alloc<T>(dx);
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dx, static_cast<T>(0));

    int64_t output_grad_size = out_grad.numel();
    uint32_t blocks = NumBlocks(output_grad_size);
    uint32_t threads = kNumCUDAThreads;

    if (output_grad_size > 0) {
      if (output_grad_size > std::numeric_limits<int32_t>::max() ||
          dx->numel() > std::numeric_limits<int32_t>::max()) {
        GPURoiPoolBackward<T, int64_t>
            <<<blocks, threads, 0, dev_ctx.stream()>>>(output_grad_size,
                                                       boxes.data<T>(),
                                                       out_grad.data<T>(),
                                                       arg_max.data<int64_t>(),
                                                       rois_num,
                                                       spatial_scale,
                                                       channels,
                                                       height,
                                                       width,
                                                       pooled_height,
                                                       pooled_width,
                                                       roi_id_data,
                                                       dx->data<T>());
      } else {
        GPURoiPoolBackward<T, int32_t>
            <<<blocks, threads, 0, dev_ctx.stream()>>>(output_grad_size,
                                                       boxes.data<T>(),
                                                       out_grad.data<T>(),
                                                       arg_max.data<int64_t>(),
                                                       rois_num,
                                                       spatial_scale,
                                                       channels,
                                                       height,
                                                       width,
                                                       pooled_height,
                                                       pooled_width,
                                                       roi_id_data,
                                                       dx->data<T>());
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_pool_grad, GPU, ALL_LAYOUT, phi::RoiPoolGradKernel, float, double) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT64);
}
