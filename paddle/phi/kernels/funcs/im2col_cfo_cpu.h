/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

/**
 * The most common im2col algorithm.
 * Support dilation, stride and padding.
 */
template <typename T>
inline void im2col_common(const phi::DenseTensor& im,
                          const std::vector<int>& dilation,
                          const std::vector<int>& stride,
                          const std::vector<int>& padding,
                          phi::DenseTensor* col,
                          const DataLayout data_layout = DataLayout::NCHW) {
  int64_t im_channels =
      (data_layout != DataLayout::NHWC ? im.dims()[0] : im.dims()[2]);
  int64_t im_height =
      (data_layout != DataLayout::NHWC ? im.dims()[1] : im.dims()[0]);
  int64_t im_width =
      (data_layout != DataLayout::NHWC ? im.dims()[2] : im.dims()[1]);
  int64_t filter_height = col->dims()[1];
  int64_t filter_width = col->dims()[2];
  int64_t output_height = col->dims()[3];
  int64_t output_width = col->dims()[4];

  int64_t channels_col = im_channels * filter_height * filter_width;

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % filter_width;
    int64_t h_offset = (c / filter_width) % filter_height;
    int64_t c_im = c / (filter_width * filter_height);
    for (int64_t h = 0; h < output_height; ++h) {
      int64_t im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
      for (int64_t w = 0; w < output_width; ++w) {
        int64_t im_col_idx =
            w * stride[1] - padding[1] + w_offset * dilation[1];

        // Calculate col_idx using 64-bit arithmetic to prevent overflow
        int64_t col_idx64 = (c * output_height + h) * output_width + w;

        // Check bounds first to avoid buffer overflow in im_idx calculation
        if (im_row_idx < 0 || im_row_idx >= im_height || im_col_idx < 0 ||
            im_col_idx >= im_width) {
          *(col_data + col_idx64) = static_cast<T>(0);
        } else {
          int64_t im_idx64;
          if (data_layout != DataLayout::NHWC) {
            im_idx64 = (c_im * im_height + im_row_idx) * im_width + im_col_idx;
          } else {
            im_idx64 =
                (im_row_idx * im_width + im_col_idx) * im_channels + c_im;
          }
          *(col_data + col_idx64) = *(im_data + im_idx64);
        }
      }
    }
  }
}

/**
 * im2col algorithm with strides == 1, dilations == 1, paddings == 0
 */
template <typename T>
inline void im2col_sh1sw1dh1dw1ph0pw0(
    const phi::DenseTensor& im,
    phi::DenseTensor* col,
    const DataLayout data_layout = DataLayout::NCHW) {
  int im_channels =
      (data_layout != DataLayout::NHWC ? im.dims()[0] : im.dims()[2]);
  int im_height =
      (data_layout != DataLayout::NHWC ? im.dims()[1] : im.dims()[0]);
  int im_width =
      (data_layout != DataLayout::NHWC ? im.dims()[2] : im.dims()[1]);
  int64_t filter_height = col->dims()[1];
  int64_t filter_width = col->dims()[2];
  int64_t output_height = col->dims()[3];
  int64_t output_width = col->dims()[4];

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  int64_t col_matrix_width = output_width * output_height;
  int64_t im_size = im_height * im_width;
  size_t copy_size = sizeof(T) * output_width;
  const T* im_data_oh = im_data;
  T* dst_data_oh = col_data;
  for (int64_t oh = 0; oh < output_height; ++oh) {
    const T* src_data_ic = im_data_oh;
    T* dst_data = dst_data_oh;
    for (int ic = 0; ic < im_channels; ++ic) {
      const T* src_data = src_data_ic;
      for (int64_t kh = 0; kh < filter_height; ++kh) {
        for (int64_t kw = 0; kw < filter_width; ++kw) {
          if (data_layout != DataLayout::NHWC) {
            std::memcpy(dst_data, src_data + kw, copy_size);
          } else {
            for (int64_t kow = 0; kow < output_width; ++kow) {
              dst_data[kow] =
                  im_data[((oh + kh) * im_width + kw + kow) * im_channels + ic];
            }
          }
          dst_data = dst_data + col_matrix_width;
        }
        src_data = src_data + im_width;
      }
      src_data_ic = src_data_ic + im_size;
    }
    im_data_oh = im_data_oh + im_width;
    dst_data_oh = dst_data_oh + output_width;
  }
}

/**
 * im2col algorithm with strides == 1, dilations == 1, paddings == 1
 * and filter_width == 1 have a special implementation
 */
template <typename T>
inline void im2col_sh1sw1dh1dw1ph1pw1(const phi::DenseTensor& im,
                                      phi::DenseTensor* col,
                                      const DataLayout data_layout) {
  int im_channels =
      (data_layout != DataLayout::NHWC ? im.dims()[0] : im.dims()[2]);
  int im_height =
      (data_layout != DataLayout::NHWC ? im.dims()[1] : im.dims()[0]);
  int im_width =
      (data_layout != DataLayout::NHWC ? im.dims()[2] : im.dims()[1]);
  int64_t filter_height = col->dims()[1];
  int64_t filter_width = col->dims()[2];
  int64_t output_height = col->dims()[3];
  int64_t output_width = col->dims()[4];

  constexpr int plh = 1;
  constexpr int prh = 1;
  constexpr int plw = 1;
  constexpr int prw = 1;

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  int64_t im_size = im_height * im_width;
  int64_t col_matrix_width = output_width * output_height;
  int64_t col_block_fh = filter_width * col_matrix_width;  // fw*oh*ow
  int64_t col_block_ic = filter_height * col_block_fh;     // fh*fw*oh*ow

  // fill height padding
  {
    size_t copy_size = sizeof(T) * output_width;
    T* col_start_l = col_data;
    T* col_start_r = col_data + (filter_height - 1) * col_block_fh +
                     col_matrix_width - output_width;
    for (int ic = 0; ic < im_channels; ++ic) {
      T* dst_data_l = col_start_l;
      T* dst_data_r = col_start_r;
      for (int64_t kw = 0; kw < filter_width; ++kw) {
        std::memset(dst_data_l, 0, copy_size);
        std::memset(dst_data_r, 0, copy_size);
        dst_data_l = dst_data_l + col_matrix_width;
        dst_data_r = dst_data_r + col_matrix_width;
      }
      col_start_l = col_start_l + col_block_ic;
      col_start_r = col_start_r + col_block_ic;
    }
  }

  auto pad = static_cast<T>(0);
  if (filter_width == 1) {
    // fill width padding
    T* dst_data_ic = col_data;
    for (int ic = 0; ic < im_channels; ++ic) {
      T* dst_data_kh = dst_data_ic;
      for (int64_t kh = 0; kh < filter_height; ++kh) {
        T* dst_data = dst_data_kh;
        for (int64_t oh = 0; oh < output_height; ++oh) {
          *dst_data = pad;
          dst_data = dst_data + output_width - 1;
          *dst_data = pad;
          ++dst_data;
        }
        dst_data_kh = dst_data_kh + col_block_fh;
      }
      dst_data_ic = dst_data_ic + col_block_ic;
    }
    // fill core
    for (int64_t oh = 0; oh < output_height; ++oh) {
      const T* im_data_start =
          im_data + (oh - plh > 0 ? oh - plh : 0) * im_width;
      T* dst_data = col_data + oh * output_width;
      for (int ic = 0; ic < im_channels; ++ic) {
        const T* src_data = im_data_start + ic * im_size;
        for (int kh = 0; kh < filter_height; ++kh) {
          if ((oh < plh && kh < plh) || (oh > (output_height - prh - 1) &&
                                         kh > (filter_height - prh - 1))) {
            dst_data = dst_data + col_matrix_width;
            continue;
          }
          if (data_layout != DataLayout::NHWC) {
            // Safe memcpy for filter_width == 1 case
            int want = output_width - plw - prw;
            int avail = im_width;
            int n = std::max(0, std::min(want, avail));
            if (n > 0) {
              std::memcpy(dst_data + plw, src_data, sizeof(T) * n);
            }
            // Zero any shortfall
            int shortfall = want - n;
            if (shortfall > 0) {
              std::memset(dst_data + plw + n, 0, sizeof(T) * shortfall);
            }
          } else {
            for (int kow = 0; kow < output_width - plw - prw; ++kow) {
              int im_row = oh - plh + kh;
              int im_col = kow;
              if (im_row >= 0 && im_row < im_height && im_col >= 0 &&
                  im_col < im_width) {
                dst_data[plw + kow] =
                    im_data[(im_row * im_width + im_col) * im_channels + ic];
              } else {
                dst_data[plw + kow] = static_cast<T>(0);
              }
            }
          }
          dst_data = dst_data + col_matrix_width;
          src_data = src_data + im_width;
        }
      }
    }
    return;
  }

  // filter_width != 1
  // fill width padding
  T* dst_data_ic = col_data;
  for (int ic = 0; ic < im_channels; ++ic) {
    T* dst_data_kh = dst_data_ic;
    for (int kh = 0; kh < filter_height; ++kh) {
      for (T* dst_data :
           {dst_data_kh,
            dst_data_kh + (filter_width - prw) * col_matrix_width +
                output_width - 1}) {
        // TODO(TJ): from plh, saving repeated assignment
        for (int oh = 0; oh < output_height; ++oh) {
          *dst_data = pad;
          dst_data = dst_data + output_width;
        }
      }
      dst_data_kh = dst_data_kh + col_block_fh;
    }
    dst_data_ic = dst_data_ic + col_block_ic;
  }

  // TODO(TJ): use array like: size_t copy_size[kw]={sizeof(T) *
  // (output_width-1)}
  // length of copy_size is equal kw.
  for (int oh = 0; oh < output_height; ++oh) {
    const T* im_data_start = im_data + (oh - plh > 0 ? oh - plh : 0) * im_width;
    T* dst_data = col_data + oh * output_width;
    for (int ic = 0; ic < im_channels; ++ic) {
      const T* src_data = im_data_start + ic * im_size;
      for (int kh = 0; kh < filter_height; ++kh) {
        if ((oh < plh && kh < plh) || (oh > (output_height - prh - 1) &&
                                       kh > (filter_height - prh - 1))) {
          dst_data = dst_data + filter_width * col_matrix_width;
          continue;
        }
        // TODO(TJ): reuse plw-kw outside this for
        // try to unify
        for (int kw = 0; kw < plw; ++kw) {
          if (data_layout != DataLayout::NHWC) {
            // Left band: clamp memcpy to avoid over-read
            int want = output_width - (plw - kw);
            int src_col_start = 0;
            int avail = im_width - src_col_start;
            int n = std::max(0, std::min(want, avail));
            if (n > 0) {
              std::memcpy(dst_data + (plw - kw),
                          src_data + src_col_start,
                          sizeof(T) * n);
            }
            // Zero any shortfall
            int shortfall = want - n;
            if (shortfall > 0) {
              std::memset(dst_data + (plw - kw) + n, 0, sizeof(T) * shortfall);
            }
          } else {
            for (int kow = 0; kow < output_width - (plw - kw); ++kow) {
              int im_row = oh - plh + kh;
              int im_col = kow;
              if (im_row >= 0 && im_row < im_height && im_col >= 0 &&
                  im_col < im_width) {
                dst_data[plw - kw + kow] =
                    im_data[(im_row * im_width + im_col) * im_channels + ic];
              } else {
                dst_data[plw - kw + kow] = static_cast<T>(0);
              }
            }
          }
          dst_data = dst_data + col_matrix_width;
        }
        for (int kw = plw; kw < filter_width - prw; ++kw) {
          if (data_layout != DataLayout::NHWC) {
            // Middle band: clamp memcpy to avoid over-read
            int src_col_start = kw - plw;
            int want = output_width;
            int avail = im_width - src_col_start;
            int n = std::max(0, std::min(want, avail));
            if (n > 0) {
              std::memcpy(dst_data, src_data + src_col_start, sizeof(T) * n);
            }
            if (n < want) {
              std::memset(dst_data + n, 0, sizeof(T) * (want - n));
            }
          } else {
            for (int kow = 0; kow < output_width; ++kow) {
              int im_row = oh - plh + kh;
              int im_col = kw - plw + kow;
              if (im_row >= 0 && im_row < im_height && im_col >= 0 &&
                  im_col < im_width) {
                dst_data[kow] =
                    im_data[(im_row * im_width + im_col) * im_channels + ic];
              } else {
                dst_data[kow] = static_cast<T>(0);
              }
            }
          }
          dst_data = dst_data + col_matrix_width;
        }
        int i = 1;
        for (int kw = filter_width - prw; kw < filter_width; ++kw, ++i) {
          if (data_layout != DataLayout::NHWC) {
            // Right band: clamp memcpy to avoid over-read
            int src_col_start = kw - plw;
            int want = output_width - i;
            int avail = im_width - src_col_start;
            int n = std::max(0, std::min(want, avail));
            if (n > 0) {
              std::memcpy(dst_data, src_data + src_col_start, sizeof(T) * n);
            }
            if (n < want) {
              std::memset(dst_data + n, 0, sizeof(T) * (want - n));
            }
          } else {
            for (int kow = 0; kow < output_width - i; ++kow) {
              int im_row = oh - plh + kh;
              int im_col = kw - plw + kow;
              if (im_row >= 0 && im_row < im_height && im_col >= 0 &&
                  im_col < im_width) {
                dst_data[kow] =
                    im_data[(im_row * im_width + im_col) * im_channels + ic];
              } else {
                dst_data[kow] = static_cast<T>(0);
              }
            }
          }
          dst_data = dst_data + col_matrix_width;
        }
        src_data = src_data + im_width;
      }
    }
  }
}

}  // namespace funcs
}  // namespace phi
