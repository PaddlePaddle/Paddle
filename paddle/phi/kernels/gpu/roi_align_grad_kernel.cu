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

#include "paddle/phi/kernels/roi_align_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;
static constexpr int kROISize = 4;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <class T>
__device__ void BilinearInterpolateGradient(const int height,
                                            const int width,
                                            T y,
                                            T x,
                                            T* w1,
                                            T* w2,
                                            T* w3,
                                            T* w4,
                                            int* x_low,
                                            int* x_high,
                                            int* y_low,
                                            int* y_high) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return;
  }

  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);
  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = static_cast<T>(*y_low);
  } else {
    *y_high = *y_low + 1;
  }
  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = static_cast<T>(*x_low);
  } else {
    *x_high = *x_low + 1;
  }
  T ly = y - *y_low, lx = x - *x_low;
  T hy = 1. - ly, hx = 1. - lx;
  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <typename T>
__global__ void GPURoiAlignBackward(const int nthreads,
                                    const T* input_rois,
                                    const T* out_grad,
                                    const int num_rois,
                                    const float spatial_scale,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int sampling_ratio,
                                    int* roi_batch_id_data,
                                    T* input_grad,
                                    const bool continuous_coordinate) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;
    const T* offset_input_rois = input_rois + n * kROISize;
    int roi_batch_ind = roi_batch_id_data[n];

    T roi_offset = continuous_coordinate ? T(0.5) : 0;
    T roi_xmin = offset_input_rois[0] * spatial_scale - roi_offset;
    T roi_ymin = offset_input_rois[1] * spatial_scale - roi_offset;
    T roi_xmax = offset_input_rois[2] * spatial_scale - roi_offset;
    T roi_ymax = offset_input_rois[3] * spatial_scale - roi_offset;

    T roi_width = roi_xmax - roi_xmin;
    T roi_height = roi_ymax - roi_ymin;
    if (!continuous_coordinate) {
      roi_width = max(roi_width, static_cast<T>(1.));
      roi_height = max(roi_height, static_cast<T>(1.));
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_input_grad =
        input_grad + (roi_batch_ind * channels + c) * height * width;

    const T* offset_out_grad =
        out_grad + (n * channels + c) * pooled_height * pooled_width;
    const T out_grad_this_bin = offset_out_grad[ph * pooled_width + pw];

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T count = roi_bin_grid_h * roi_bin_grid_w;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_ymin + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_xmin + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T w1 = 0, w2 = 0, w3 = 0, w4 = 0;
        int x_low = -1, x_high = -1, y_low = -1, y_high = -1;
        BilinearInterpolateGradient(height,
                                    width,
                                    y,
                                    x,
                                    &w1,
                                    &w2,
                                    &w3,
                                    &w4,
                                    &x_low,
                                    &x_high,
                                    &y_low,
                                    &y_high);
        T diff1 = out_grad_this_bin * w1 / count;
        T diff2 = out_grad_this_bin * w2 / count;
        T diff3 = out_grad_this_bin * w3 / count;
        T diff4 = out_grad_this_bin * w4 / count;
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          phi::CudaAtomicAdd(offset_input_grad + y_low * width + x_low, diff1);
          phi::CudaAtomicAdd(offset_input_grad + y_low * width + x_high, diff2);
          phi::CudaAtomicAdd(offset_input_grad + y_high * width + x_low, diff3);
          phi::CudaAtomicAdd(offset_input_grad + y_high * width + x_high,
                             diff4);
        }
      }
    }
  }
}

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& boxes,
                        const paddle::optional<DenseTensor>& boxes_num,
                        const DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        DenseTensor* dx) {
  int rois_num = boxes.dims()[0];
  int channels = x.dims()[1];
  int height = x.dims()[2];
  int width = x.dims()[3];

  if (!dx) {
    return;
  }

  DenseTensor box_batch_id_list;
  box_batch_id_list.Resize({rois_num});
  int* box_batch_size = dev_ctx.template HostAlloc<int>(&box_batch_id_list);

  auto cplace = phi::CPUPlace();
  auto gplace = dev_ctx.GetPlace();
  if (boxes_num) {
    int boxes_batch_size = boxes_num->numel();
    if (boxes_num->dtype() == phi::DataType::INT64) {
      std::vector<int64_t> boxes_num_list(boxes_batch_size);
      memory_utils::Copy(cplace,
                         boxes_num_list.data(),
                         gplace,
                         boxes_num->data<int64_t>(),
                         sizeof(int64_t) * boxes_batch_size,
                         0);
      int64_t start = 0;
      for (int64_t n = 0; n < boxes_batch_size; ++n) {
        for (int64_t i = start; i < start + boxes_num_list[n]; ++i) {
          box_batch_size[i] = n;
        }
        start += boxes_num_list[n];
      }
    } else if (boxes_num->dtype() == phi::DataType::INT32) {
      std::vector<int> boxes_num_list(boxes_batch_size);
      memory_utils::Copy(cplace,
                         boxes_num_list.data(),
                         gplace,
                         boxes_num->data<int>(),
                         sizeof(int) * boxes_batch_size,
                         0);
      int start = 0;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (size_t i = start; i < start + boxes_num_list[n]; ++i) {
          box_batch_size[i] = n;
        }
        start += boxes_num_list[n];
      }
    }
  } else {
    auto boxes_lod = boxes.lod().back();
    int boxes_batch_size = boxes_lod.size() - 1;
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_size[i] = n;
      }
    }
  }
  auto roi_ptr = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      box_batch_id_list.numel() * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
  int bytes = box_batch_id_list.numel() * sizeof(int);
  memory_utils::Copy(
      gplace, roi_id_data, cplace, box_batch_size, bytes, dev_ctx.stream());
  dev_ctx.template Alloc<T>(dx);

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, dx, static_cast<T>(0));

  int output_grad_size = out_grad.numel();
  int blocks = NumBlocks(output_grad_size);
  int threads = kNumCUDAThreads;

  if (output_grad_size > 0) {
    GPURoiAlignBackward<T>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(output_grad_size,
                                                   boxes.data<T>(),
                                                   out_grad.data<T>(),
                                                   rois_num,
                                                   spatial_scale,
                                                   channels,
                                                   height,
                                                   width,
                                                   pooled_height,
                                                   pooled_width,
                                                   sampling_ratio,
                                                   roi_id_data,
                                                   dx->data<T>(),
                                                   aligned);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_align_grad, GPU, ALL_LAYOUT, phi::RoiAlignGradKernel, float, double) {}
