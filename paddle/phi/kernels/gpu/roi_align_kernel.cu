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

#include "paddle/phi/kernels/roi_align_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;
static constexpr int kROISize = 4;

static inline uint32_t NumBlocks(const int64_t N) {
  return static_cast<uint32_t>(
      std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
               static_cast<int64_t>(kNumMaximumNumBlocks)));
}

template <class T, typename IndexType>
__device__ T BilinearInterpolate(const T* input_data,
                                 const IndexType height,
                                 const IndexType width,
                                 T y,
                                 T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }
  y = y <= 0 ? 0 : y;
  x = x <= 0 ? 0 : x;
  IndexType y_low = static_cast<IndexType>(y);
  IndexType x_low = static_cast<IndexType>(x);
  IndexType y_high;
  IndexType x_high;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }
  T ly = y - y_low, lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  T v1 = input_data[y_low * width + x_low];
  T v2 = input_data[y_low * width + x_high];
  T v3 = input_data[y_high * width + x_low];
  T v4 = input_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <class T, typename IndexType>
__global__ void GPURoiAlignForward(const IndexType nthreads,
                                   const T* input_data,
                                   const T* input_rois,
                                   const float spatial_scale,
                                   const IndexType channels,
                                   const IndexType height,
                                   const IndexType width,
                                   const int pooled_height,
                                   const int pooled_width,
                                   const int sampling_ratio,
                                   int* roi_batch_id_data,
                                   T* output_data,
                                   const bool continuous_coordinate) {
  CUDA_KERNEL_LOOP_TYPE(i, nthreads, IndexType) {
    IndexType pw = i % pooled_width;
    IndexType ph = (i / pooled_width) % pooled_height;
    IndexType c = (i / pooled_width / pooled_height) % channels;
    IndexType n = i / pooled_width / pooled_height / channels;

    const T* offset_input_rois = input_rois + n * kROISize;
    int roi_batch_ind = roi_batch_id_data[n];

    T roi_offset = continuous_coordinate ? static_cast<T>(0.5) : 0;
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

    const T* offset_input_data =
        input_data + (roi_batch_ind * channels + c) * height * width;

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    T output_val = 0;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_ymin + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_xmin + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        T val = BilinearInterpolate<T, IndexType>(
            offset_input_data, height, width, y, x);
        output_val += val;
      }
    }
    output_val /= count;
    output_data[i] = output_val;
  }
}

template <typename T, typename Context>
void RoiAlignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& boxes,
                    const paddle::optional<DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  if (x.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), 0, out);
    return;
  }
  auto in_dims = x.dims();
  int64_t batch_size = in_dims[0];
  int64_t channels = in_dims[1];
  int64_t height = in_dims[2];
  int64_t width = in_dims[3];

  int64_t rois_num = boxes.dims()[0];

  if (rois_num == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  int64_t output_size = out->numel();
  uint32_t blocks = NumBlocks(output_size);
  uint32_t threads = kNumCUDAThreads;
#ifdef WITH_NV_JETSON
  backends::gpu::ChangeThreadNum(dev_ctx, &threads, 256);
#endif
  DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize({rois_num});
  int* roi_batch_id_data = dev_ctx.template HostAlloc<int>(&roi_batch_id_list);
  auto cplace = phi::CPUPlace();
  auto gplace = dev_ctx.GetPlace();
  if (boxes_num) {
    int64_t boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The boxes_batch_size and imgs "
            "batch_size must be the same. But received boxes_batch_size = %d, "
            "batch_size = %d",
            boxes_batch_size,
            batch_size));

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
          roi_batch_id_data[i] = n;
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
      int64_t start = 0;
      for (int64_t n = 0; n < boxes_batch_size; ++n) {
        for (int64_t i = start; i < start + boxes_num_list[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += boxes_num_list[n];
      }
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(lod.empty(),
                      false,
                      errors::InvalidArgument("Input(ROIs) in ROIAlignOp does "
                                              "not contain LoD information."));
    auto boxes_lod = lod.back();
    int64_t boxes_batch_size = boxes_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The batch size of rois and batch size "
            "of images must be the same. But received rois batch size = %d, "
            "and images batch size = %d",
            boxes_batch_size,
            batch_size));
    int64_t boxes_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(
        rois_num,
        boxes_num_with_lod,
        errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            rois_num,
            boxes_num_with_lod));
    for (int64_t n = 0; n < boxes_batch_size; ++n) {
      for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }
  int64_t bytes = roi_batch_id_list.numel() * sizeof(int);
  auto roi_ptr = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
  memory_utils::Copy(
      gplace, roi_id_data, cplace, roi_batch_id_data, bytes, dev_ctx.stream());
  if (output_size > std::numeric_limits<int>::max() ||
      x.numel() > std::numeric_limits<int>::max()) {
    GPURoiAlignForward<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size,
        x.data<T>(),
        boxes.data<T>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        roi_id_data,
        dev_ctx.template Alloc<T>(out),
        aligned);
  } else {
    GPURoiAlignForward<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size,
        x.data<T>(),
        boxes.data<T>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        roi_id_data,
        dev_ctx.template Alloc<T>(out),
        aligned);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_align, GPU, ALL_LAYOUT, phi::RoiAlignKernel, float, double) {}
