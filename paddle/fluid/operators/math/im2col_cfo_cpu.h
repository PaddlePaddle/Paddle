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

#pragma once

#include <vector>
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * The most common im2col algorithm.
 * Support dilation, stride and padding.
 */
template <typename T>
inline void im2col_common(const framework::Tensor& im,
                          const std::vector<int>& dilation,
                          const std::vector<int>& stride,
                          const std::vector<int>& padding,
                          framework::Tensor* col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int output_height = col->dims()[3];
  int output_width = col->dims()[4];
  int channels_col = im_channels * filter_height * filter_width;

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % filter_width;
    int h_offset = (c / filter_width) % filter_height;
    int c_im = c / (filter_width * filter_height);
    for (int h = 0; h < output_height; ++h) {
      int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
      for (int w = 0; w < output_width; ++w) {
        int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
        int col_idx = (c * output_height + h) * output_width + w;
        int im_idx = (im_row_idx + c_im * im_height) * im_width + im_col_idx;
        col_data[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                             im_col_idx < 0 || im_col_idx >= im_width)
                                ? static_cast<T>(0)
                                : im_data[im_idx];
      }
    }
  }
}

/*
 * im2col algorithm with strides == 1, dilations == 1, paddings == 0
 * */
template <typename T>
inline void im2col_sh1sw1dh1dw1ph0pw0(const framework::Tensor& im,
                                      framework::Tensor* col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int output_height = col->dims()[3];
  int output_width = col->dims()[4];

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  int col_matrix_width = output_width * output_height;
  int im_size = im_height * im_width;
  size_t copy_size = sizeof(T) * output_width;
  for (int oh = 0; oh < output_height; ++oh) {
    const T* im_data_start = im_data + oh * im_width;
    T* dst_data = col_data + oh * output_width;
    for (int ic = 0; ic < im_channels; ++ic) {
      const T* src_data = im_data_start + ic * im_size;
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          std::memcpy(dst_data, src_data + kw, copy_size);
          dst_data = dst_data + col_matrix_width;
        }
        src_data = src_data + im_width;
      }
    }
  }
}

// further optimize: padding == 1 need special
template <typename T>
inline void im2col_sh1sw1dh1dw1(const framework::Tensor& im,
                                const std::vector<int>& padding,
                                framework::Tensor* col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int output_height = col->dims()[3];
  int output_width = col->dims()[4];
  const int sh = 1;
  const int sw = 1;

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  int col_matrix_width = output_width * output_height;
  int im_size = im_height * im_width;

  int plh = padding[0];
  int plw = padding[1];
  int prh = (output_height - 1) * sh + filter_height - im_height - plh;
  int prw = (output_width - 1) * sw + filter_width - im_width - plw;

  // fill height padding : 0 ~ plh-1, (oh-prh) ~ (oh-1)
  // TODO(TJ): refine ph*xxx
  assert(plh == prh);                                  // because stride_h == 1
  int col_block_fh = filter_width * col_matrix_width;  // fw*oh*ow
  int col_block_ic = filter_height * col_block_fh;     // fh*fw*oh*ow
  for (int ph = 0; ph < plh; ++ph) {
    int sz = output_width * (plh - ph);
    size_t copy_sz = sizeof(T) * sz;
    T* col_start_l = col_data + ph * col_block_fh;
    T* col_start_r = col_data + (filter_height - ph - 1) * col_block_fh +
                     col_matrix_width - sz;
    for (int ic = 0; ic < im_channels; ++ic) {
      T* dst_data_l = col_start_l + ic * col_block_ic;
      T* dst_data_r = col_start_r + ic * col_block_ic;
      for (int kw = 0; kw < filter_width; ++kw) {
        std::memset(dst_data_l, 0, copy_sz);
        std::memset(dst_data_r, 0, copy_sz);
        dst_data_l = dst_data_l + col_matrix_width;
        dst_data_r = dst_data_r + col_matrix_width;
      }
    }
  }

  // fill width padding
  assert(plw == prw);  // because stride_w == 1
  if (plw == 1) {
    auto pad = static_cast<T>(0);  // padding zero
    for (int ic = 0; ic < im_channels; ++ic) {
      // TODO(TJ): use add and resue stride
      T* dst_data_ic = col_data + ic * col_block_ic;
      for (int kh = 0; kh < filter_height; ++kh) {
        T* dst_data_kh = dst_data_ic + kh * col_block_fh;
        for (T* dst_data :
             {dst_data_kh, dst_data_kh +
                               (filter_width - prw) * col_matrix_width +
                               output_width - 1}) {
          // TODO(TJ): from plh, saving repeated assignment
          for (int oh = 0; oh < output_height; ++oh) {
            *dst_data = pad;
            dst_data = dst_data + output_width;
          }
        }
      }
    }
  } else {
    // padding_size > 1
    for (int ic = 0; ic < im_channels; ++ic) {
      // TODO(TJ): use add and resue stride
      T* dst_data_ic = col_data + ic * col_block_ic;
      for (int kh = 0; kh < filter_height; ++kh) {
        T* dst_data_kh = dst_data_ic + kh * col_block_fh;
        for (int kw = 0; kw < plw; ++kw) {
          // TODO(TJ): reuse array outside this for
          size_t sz = sizeof(T) * (plw - kw);
          T* dst_data = dst_data_kh + kw * col_matrix_width;
          // TODO(TJ): from plh, saving repeated assignment
          for (int oh = 0; oh < output_height; ++oh) {
            std::memset(dst_data, 0, sz);
            dst_data = dst_data + output_width;
          }
        }
        // TODO(TJ): use reverse to save cache
        for (int kw = 0; kw < prw; ++kw) {
          // TODO(TJ): reuse array outside this for
          auto num = (prw - kw);
          size_t sz = sizeof(T) * num;
          T* dst_data = dst_data_kh +
                        (filter_width - 1 - kw) * col_matrix_width +
                        output_width - num;
          // TODO(TJ): from plh, saving repeated assignment
          for (int oh = 0; oh < output_height; ++oh) {
            std::memset(dst_data, 0, sz);
            dst_data = dst_data + output_width;
          }
        }
      }
    }
  }

  // fill im_data
  // padding cover two cases:
  // 1. kw > 2*pw: kw = 3, pw = 1
  // 0 x x x x ... x x x x 0
  // 1 1 1             1 1 1
  // ==>
  // 0 x ... x x
  // x x ... x x
  // x x ... x 0
  // 2. kw < 2*pw: kw = 3, pw = 2
  // 0 0 x x x ... x x x 0 0
  // 1 1 1             1 1 1
  // ==>
  // 0 0 x ... x x x
  // 0 x x ... x x 0
  // x x x ... x 0 0

  // TODO(TJ): use array like: size_t copy_size[kw]={sizeof(T) *
  // (output_width-1)}
  // length of copy_size is equal kw.
  if (plw + prw < filter_width) {
    for (int oh = 0; oh < output_height; ++oh) {
      const T* im_data_start =
          im_data + (oh - plh > 0 ? oh - plh : 0) * im_width;
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
            std::memcpy(dst_data + (plw - kw), src_data,
                        sizeof(T) * (output_width - (plw - kw)));
            dst_data = dst_data + col_matrix_width;
          }
          for (int kw = plw; kw < filter_width - prw; ++kw) {
            std::memcpy(dst_data, src_data + (kw - plw),
                        sizeof(T) * output_width);
            dst_data = dst_data + col_matrix_width;
          }
          int i = 1;
          for (int kw = filter_width - prw; kw < filter_width; ++kw, ++i) {
            std::memcpy(dst_data, src_data + (kw - plw),
                        sizeof(T) * (output_width - i));
            dst_data = dst_data + col_matrix_width;
          }
          src_data = src_data + im_width;
        }
      }
    }
  } else {
    LOG(FATAL) << "Not implement yet";
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
