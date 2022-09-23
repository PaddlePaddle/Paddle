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

#include "paddle/phi/kernels/roi_pool_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void GPURoiPoolForward(const int nthreads,
                                  const T* input_data,
                                  const T* input_rois,
                                  const float spatial_scale,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int pooled_height,
                                  const int pooled_width,
                                  int* box_batch_id_data,
                                  T* output_data,
                                  int64_t* arg_max_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    const T* offset_input_rois = input_rois + n * kROISize;
    int box_batch_ind = box_batch_id_data[n];
    int box_start_w = round(offset_input_rois[0] * spatial_scale);
    int box_start_h = round(offset_input_rois[1] * spatial_scale);
    int box_end_w = round(offset_input_rois[2] * spatial_scale);
    int box_end_h = round(offset_input_rois[3] * spatial_scale);

    int box_width = max(box_end_w - box_start_w + 1, 1);
    int box_height = max(box_end_h - box_start_h + 1, 1);

    int hstart = static_cast<int>(
        floor(static_cast<double>(ph) * static_cast<double>(box_height) /
              static_cast<double>(pooled_height)));
    int wstart = static_cast<int>(
        floor(static_cast<double>(pw) * static_cast<double>(box_width) /
              static_cast<double>(pooled_width)));
    int hend = static_cast<int>(
        ceil(static_cast<double>(ph + 1) * static_cast<double>(box_height) /
             static_cast<double>(pooled_height)));
    int wend = static_cast<int>(
        ceil(static_cast<double>(pw + 1) * static_cast<double>(box_width) /
             static_cast<double>(pooled_width)));
    hstart = min(max(hstart + box_start_h, 0), height);
    hend = min(max(hend + box_start_h, 0), height);
    wstart = min(max(wstart + box_start_w, 0), width);
    wend = min(max(wend + box_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    T maxval = is_empty ? 0 : -std::numeric_limits<T>::max();
    int maxidx = -1;
    const T* offset_input_data =
        input_data + (box_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_data_index = h * width + w;
        if (offset_input_data[input_data_index] > maxval) {
          maxval = offset_input_data[input_data_index];
          maxidx = input_data_index;
        }
      }
    }
    output_data[i] = maxval;
    if (arg_max_data) {
      arg_max_data[i] = maxidx;
    }
  }
}

template <typename T, typename Context>
void RoiPoolKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& boxes,
                   const paddle::optional<DenseTensor>& boxes_num,
                   int pooled_height,
                   int pooled_width,
                   float spatial_scale,
                   DenseTensor* out,
                   DenseTensor* arg_max) {
  auto x_dims = x.dims();
  int batch_size = x_dims[0];
  auto in_stride = phi::stride(x_dims);
  int channels = x_dims[1];
  int height = x_dims[2];
  int width = x_dims[3];

  int rois_num = boxes.dims()[0];

  if (rois_num == 0) return;

  int output_size = out->numel();
  int blocks = NumBlocks(output_size);
  int threads = kNumCUDAThreads;

  DenseTensor box_batch_id_list;
  box_batch_id_list.Resize({rois_num});
  int* box_batch_id_data = dev_ctx.template HostAlloc<int>(&box_batch_id_list);
  auto gplace = dev_ctx.GetPlace();

  if (boxes_num) {
    int boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The batch size of input(ROIs) and input(X) must be the same but "
            "received batch size of input(ROIs) and input(X) is %d and %d "
            "respectively.",
            boxes_batch_size,
            batch_size));
    std::vector<int> boxes_num_list(boxes_batch_size);
    paddle::memory::Copy(phi::CPUPlace(),
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
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The batch size of input(ROIs) and input(X) must be the same but "
            "received batch size of input(ROIs) and input(X) is %d and %d "
            "respectively.",
            boxes_batch_size,
            batch_size));

    int boxes_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(rois_num,
                      boxes_num_with_lod,
                      phi::errors::InvalidArgument(
                          "The number of rois from input(ROIs) and its LOD "
                          "must be the same. Received rois %d of input(ROIs) "
                          "but the number of rois %d from its LOD is %d",
                          rois_num,
                          boxes_num_with_lod));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_id_data[i] = n;
      }
    }
  }

  int bytes = box_batch_id_list.numel() * sizeof(int);
  auto box_ptr = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* box_id_data = reinterpret_cast<int*>(box_ptr->ptr());
  paddle::memory::Copy(gplace,
                       box_id_data,
                       phi::CPUPlace(),
                       box_batch_id_data,
                       bytes,
                       dev_ctx.stream());

  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* arg_max_data = dev_ctx.template Alloc<int64_t>(arg_max);

  GPURoiPoolForward<T>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(output_size,
                                                 x.data<T>(),
                                                 boxes.data<T>(),
                                                 spatial_scale,
                                                 channels,
                                                 height,
                                                 width,
                                                 pooled_height,
                                                 pooled_width,
                                                 box_id_data,
                                                 output_data,
                                                 arg_max_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_pool, GPU, ALL_LAYOUT, phi::RoiPoolKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
