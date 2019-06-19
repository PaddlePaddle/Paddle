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

/**
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
  int w_offset = 0, h_offset = 0, c_im = 0, im_row_idx;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
  for (int c = 0; c < channels_col; ++c) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        if (w == 0 && h == 0) {
          im_row_idx = -padding[0] + h_offset * dilation[0];
        }
        int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
        int col_idx = (c * output_height + h) * output_width + w;
        int im_idx = (im_row_idx + c_im * im_height) * im_width + im_col_idx;
        col_data[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                             im_col_idx < 0 || im_col_idx >= im_width)
                                ? static_cast<T>(0)
                                : im_data[im_idx];
        if (w == output_width - 1) {
          im_row_idx =
              (h + 1) * stride[0] - padding[0] + h_offset * dilation[0];
          if (h == output_height - 1) {
            w_offset = (c + 1) % filter_width;
            h_offset = ((c + 1) / filter_width) % filter_height;
            c_im = (c + 1) / (filter_width * filter_height);
          }
        }
      }
    }
  }
}

/**
 * im2col algorithm with strides == 1, dilations == 1, paddings == 0
 */
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
  const T* im_data_oh = im_data;
  T* dst_data_oh = col_data;
  T* dst_data = dst_data_oh;
  const T* src_data = im_data_oh;
  int ic_src_data_offset = im_size - filter_height * im_width;
  int oh_src_data_offset = im_width - im_channels * im_size;
  int oh_dst_data_offset =
      output_width -
      im_channels * filter_height * filter_width * col_matrix_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(4)
#endif
  for (int oh = 0; oh < output_height; ++oh) {
    for (int ic = 0; ic < im_channels; ++ic) {
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          std::memcpy(dst_data, src_data + kw, copy_size);
          dst_data += col_matrix_width;
          if (kw == filter_width - 1) {
            src_data += im_width;
            if (kh == filter_height - 1) {
              src_data += ic_src_data_offset;
              if (ic == im_channels - 1) {
                src_data += oh_src_data_offset;
                dst_data += oh_dst_data_offset;
              }
            }
          }
        }
      }
    }
  }
}

/**
 * im2col algorithm with strides == 1, dilations == 1, paddings == 1
 * and filter_width == 1 have a special implementation
 */
template <typename T>
inline void im2col_sh1sw1dh1dw1ph1pw1(const framework::Tensor& im,
                                      framework::Tensor* col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int output_height = col->dims()[3];
  int output_width = col->dims()[4];

  constexpr int plh = 1;
  constexpr int prh = 1;
  constexpr int plw = 1;
  constexpr int prw = 1;

  const T* im_data = im.data<T>();
  T* col_data = col->data<T>();
  int im_size = im_height * im_width;
  int col_matrix_width = output_width * output_height;
  int col_block_fh = filter_width * col_matrix_width;  // fw*oh*ow
  int col_block_ic = filter_height * col_block_fh;     // fh*fw*oh*ow

  // fill height padding
  {
    size_t copy_size = sizeof(T) * output_width;
    T* col_start_l = col_data;
    T* col_start_r = col_data + (filter_height - 1) * col_block_fh +
                     col_matrix_width - output_width;
    {
      T* dst_data_l = col_start_l;
      T* dst_data_r = col_start_r;
      int ic_data_offset = col_block_ic - col_matrix_width * filter_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
      for (int ic = 0; ic < im_channels; ++ic) {
        for (int kw = 0; kw < filter_width; ++kw) {
          if (kw == 0 && ic != 0) {
            dst_data_l += ic_data_offset;
            dst_data_r += ic_data_offset;
          }
          std::memset(dst_data_l, 0, copy_size);
          std::memset(dst_data_r, 0, copy_size);
          dst_data_l += col_matrix_width;
          dst_data_r += col_matrix_width;
        }
      }
    }
  }

  auto pad = static_cast<T>(0);
  if (filter_width == 1) {
    // fill width padding
    T* dst_data_ic = col_data;
    {
      T* dst_data = dst_data_ic;
      int ic_data_offset = col_block_ic - filter_height * col_block_fh -
                           output_height * output_width;
      int kh_data_offset = col_block_fh - output_height * output_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
      for (int ic = 0; ic < im_channels; ++ic) {
        for (int kh = 0; kh < filter_height; ++kh) {
          for (int oh = 0; oh < output_height; ++oh) {
            *dst_data = pad;
            *(dst_data + output_width - 1) = pad;
            dst_data += output_width;
            if (oh == output_height - 1) {
              dst_data += kh_data_offset;
              if (kh == filter_height - 1) {
                dst_data += ic_data_offset;
              }
            }
          }
        }
      }
    }
    // fill core
    size_t copy_size = sizeof(T) * (output_width - plw - prw);
    {
      const T* src_data;
      const T* im_data_start;
      T* dst_data = col_data;
      int oh_dst_data_offset =
          output_width - im_channels * filter_height * col_matrix_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
      for (int oh = 0; oh < output_height; ++oh) {
        for (int ic = 0; ic < im_channels; ++ic) {
          for (int kh = 0; kh < filter_height; ++kh) {
            if (kh == 0) {
              if (ic == 0) {
                im_data_start =
                    im_data + (oh - plh > 0 ? oh - plh : 0) * im_width;
              }
              src_data = im_data_start + ic * im_size;
            }
            if ((oh < plh && kh < plh) || (oh > (output_height - prh - 1) &&
                                           kh > (filter_height - prh - 1))) {
              continue;
            }
            std::memcpy(dst_data + plw, src_data, copy_size);
            src_data = src_data + im_width;
            dst_data += col_matrix_width;
            if (kh == filter_height - 1 && ic == im_channels - 1) {
              dst_data += oh_dst_data_offset;
            }
          }
        }
      }
    }
    return;
  }

  // filter_width != 1
  // fill width padding
  T* dst_data_ic = col_data;
  {
    T* dst_data = dst_data_ic;
    int ic_data_offset = col_block_ic - filter_height * col_block_fh;
    int kh_data_offset = col_block_fh - output_height * output_width;
    int sec_data_offset =
        (filter_width - prw) * col_matrix_width + output_width - 1;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
    for (int ic = 0; ic < im_channels; ++ic) {
      for (int kh = 0; kh < filter_height; ++kh) {
        // TODO(TJ): from plh, saving repeated assignment
        for (int oh = 0; oh < output_height; ++oh) {
          // dst_data = dst_data_ic + ic * col_block_ic + kh * col_block_fh + oh
          // * output_width;
          *dst_data = pad;
          *(dst_data + sec_data_offset) = pad;
          dst_data += output_width;
          if (oh == output_height - 1) {
            dst_data += kh_data_offset;
            if (kh == filter_height - 1) {
              dst_data += ic_data_offset;
            }
          }
        }
      }
    }
  }

  // TODO(TJ): use array like: size_t copy_size[kw]={sizeof(T) *
  // (output_width-1)}
  // length of copy_size is equal kw.
  {
    const T* src_data;
    const T* im_data_start;
    T* dst_data = col_data;
    int ic_dst_data_offset =
        output_width -
        im_channels * filter_height * filter_width * col_matrix_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
    for (int oh = 0; oh < output_height; ++oh) {
      for (int ic = 0; ic < im_channels; ++ic) {
        for (int kh = 0; kh < filter_height; ++kh) {
          if (kh == 0) {
            if (ic == 0) {
              im_data_start =
                  im_data + (oh - plh > 0 ? oh - plh : 0) * im_width;
            }
            src_data = im_data_start + ic * im_size;
          }
          if ((oh < plh && kh < plh) || (oh > (output_height - prh - 1) &&
                                         kh > (filter_height - prh - 1))) {
            dst_data = dst_data + filter_width * col_matrix_width;
            continue;
          }
          // T* dst_data = col_data + oh * output_width + (ic * filter_height +
          // kh) * filter_width * col_matrix_width;
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
          if (kh == filter_height - 1 && ic == im_channels - 1) {
            dst_data += ic_dst_data_offset;
          }
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
