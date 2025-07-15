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

#include "paddle/phi/kernels/funcs/deformable_conv_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
namespace funcs {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int64_t N) {
  return std::min<int64_t>((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                           static_cast<int64_t>(kNumMaximumNumBlocks));
}

template <typename T>
__global__ void ModulatedDeformableIm2colGpuKernel(
    const int64_t nthreads,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t channel_per_deformable_group,
    const int64_t batch_size,
    const int64_t num_channels,
    const int64_t deformable_group,
    const int64_t height_col,
    const int64_t width_col,
    T* data_col) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    const int64_t w_col = i % width_col;
    const int64_t h_col = (i / width_col) % height_col;
    const int64_t b_col = (i / width_col) / height_col % batch_size;
    const int64_t c_im = (i / width_col / height_col) / batch_size;
    const int64_t c_col = c_im * kernel_h * kernel_w;

    const int64_t deformable_group_index = c_im / channel_per_deformable_group;

    const int64_t h_in = h_col * stride_h - pad_h;
    const int64_t w_in = w_col * stride_w - pad_w;

    T* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T* data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T* data_mask_ptr =
        data_mask
            ? data_mask + (b_col * deformable_group + deformable_group_index) *
                              kernel_h * kernel_w * height_col * width_col
            : nullptr;

    for (int64_t i = 0; i < kernel_h; ++i) {
      for (int64_t j = 0; j < kernel_w; ++j) {
        const int64_t data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int64_t data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;

        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        if (data_mask_ptr) {
          const int64_t data_mask_hw_ptr =
              ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
          const T mask = data_mask_ptr[data_mask_hw_ptr];
          *data_col_ptr *= mask;
        }
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T, typename Context>
void ModulatedDeformableIm2col(const Context& dev_ctx,
                               const T* data_im,
                               const T* data_offset,
                               const T* data_mask,
                               const std::vector<int64_t>& im_shape,
                               const std::vector<int64_t>& col_shape,
                               const std::vector<int64_t>& filter_shape,
                               const std::vector<int>& paddings,
                               const std::vector<int>& strides,
                               const std::vector<int>& dilations,
                               const int deformable_groups,
                               T* data_col) {
  int64_t channel_per_deformable_group = im_shape[0] / deformable_groups;
  int64_t num_kernels =
      im_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];

  int64_t blocks = NumBlocks(num_kernels);
  int64_t threads = kNumCUDAThreads;

  ModulatedDeformableIm2colGpuKernel<T>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(num_kernels,
                                                 data_im,
                                                 data_offset,
                                                 data_mask,
                                                 im_shape[1],
                                                 im_shape[2],
                                                 filter_shape[2],
                                                 filter_shape[3],
                                                 paddings[0],
                                                 paddings[1],
                                                 strides[0],
                                                 strides[1],
                                                 dilations[0],
                                                 dilations[1],
                                                 channel_per_deformable_group,
                                                 col_shape[1],
                                                 im_shape[0],
                                                 deformable_groups,
                                                 col_shape[2],
                                                 col_shape[3],
                                                 data_col);
}

template void ModulatedDeformableIm2col(
    const phi::GPUContext& dev_ctx,
    const float* data_im,
    const float* data_offset,
    const float* data_mask,
    const std::vector<int64_t>& im_shape,
    const std::vector<int64_t>& col_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int>& paddings,
    const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const int deformable_groups,
    float* data_col);

template void ModulatedDeformableIm2col(
    const phi::GPUContext& dev_ctx,
    const double* data_im,
    const double* data_offset,
    const double* data_mask,
    const std::vector<int64_t>& im_shape,
    const std::vector<int64_t>& col_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int>& paddings,
    const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const int deformable_groups,
    double* data_col);

}  // namespace funcs
}  // namespace phi
