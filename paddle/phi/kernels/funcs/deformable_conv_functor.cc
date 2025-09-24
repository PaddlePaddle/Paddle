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

#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi::funcs {

template <typename T, typename IndexT>
inline void ModulatedDeformableIm2colCPUKernel(
    const IndexT num_kernels,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const IndexT height,
    const IndexT width,
    const IndexT kernel_h,
    const IndexT kernel_w,
    const IndexT pad_h,
    const IndexT pad_w,
    const IndexT stride_h,
    const IndexT stride_w,
    const IndexT dilation_h,
    const IndexT dilation_w,
    const IndexT channel_per_deformable_group,
    const IndexT batch_size,
    const IndexT num_channels,
    const IndexT deformable_group,
    const IndexT height_col,
    const IndexT width_col,
    T* data_col) {
  for (IndexT i = 0; i < num_kernels; i++) {
    const IndexT w_col = i % width_col;
    const IndexT h_col = (i / width_col) % height_col;
    const IndexT b_col = (i / width_col) / height_col % batch_size;
    const IndexT c_im = (i / width_col / height_col) / batch_size;
    const IndexT c_col = c_im * kernel_h * kernel_w;

    const IndexT deformable_group_index = c_im / channel_per_deformable_group;

    const IndexT h_in = h_col * stride_h - pad_h;
    const IndexT w_in = w_col * stride_w - pad_w;

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

    for (IndexT i = 0; i < kernel_h; ++i) {
      for (IndexT j = 0; j < kernel_w; ++j) {
        const IndexT data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const IndexT data_offset_w_ptr =
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
          const int data_mask_hw_ptr =
              ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
          const T mask = data_mask_ptr[data_mask_hw_ptr];
          *data_col_ptr *= mask;
        }
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T, typename Context, typename IndexT>
void ModulatedDeformableIm2col(const Context& dev_ctx UNUSED,
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

  // get outputs of im2col with offset by bilinear interpolation
  ModulatedDeformableIm2colCPUKernel<T, IndexT>(num_kernels,
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

template void ModulatedDeformableIm2col<float, phi::CPUContext, int>(
    const phi::CPUContext& dev_ctx,
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

template void ModulatedDeformableIm2col<float, phi::CPUContext, int64_t>(
    const phi::CPUContext& dev_ctx,
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

template void ModulatedDeformableIm2col<double, phi::CPUContext, int>(
    const phi::CPUContext& dev_ctx,
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

template void ModulatedDeformableIm2col<double, phi::CPUContext, int64_t>(
    const phi::CPUContext& dev_ctx,
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

}  // namespace phi::funcs
