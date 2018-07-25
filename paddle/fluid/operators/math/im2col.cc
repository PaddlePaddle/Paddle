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

#include "paddle/fluid/operators/math/im2col.h"
#include <vector>

namespace paddle {
namespace operators {
namespace math {

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                    platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& im, const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* col) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col->dims().size() == 5);

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
    // TODO(TJ): change me to template
    // further optimize: padding == 1 need special
    if (stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
        dilation[1] == 1) {
      int col_matrix_width = output_width * output_height;
      int im_size = im_height * im_width;
      if (padding[0] == 0 && padding[1] == 0) {
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
        return;
      } else {
        int plh = padding[0];
        // int plw = padding[1];
        int prh =
            (output_height - 1) * stride[0] + filter_height - im_height - plh;
        // int prw =  (output_width - 1) * stride[1] + filter_width - im_width -
        // plw;

        // fill height padding : 0 ~ plh-1, (oh-prh) ~ (oh-1)
        // TODO(TJ): refine ph*xxx
        assert(plh == prh);  // because stride_h == 1
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
        return;
      }
    }

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
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                    platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* im) {
    PADDLE_ENFORCE(im->dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int col_height = col.dims()[3];
    int col_width = col.dims()[4];

    PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2] -
                       ((dilation[0] * (filter_height - 1) + 1))) /
                              stride[0] +
                          1,
                      col_height,
                      "Output_height and padding(padding_up, padding_down) are "
                      "inconsistent.");
    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3] -
                       ((dilation[1] * (filter_width - 1) + 1))) /
                              stride[1] +
                          1,
                      col_width,
                      "Output_height and padding(padding_up, padding_down) are "
                      "inconsistent.");

    int channels_col = im_channels * filter_height * filter_width;

    T* im_data = im->data<T>();
    const T* col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int c_im = c / (filter_width * filter_height);
      for (int h = 0; h < col_height; ++h) {
        int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
        for (int w = 0; w < col_width; ++w) {
          int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
          if ((im_row_idx) >= 0 && (im_row_idx) < im_height &&
              (im_col_idx) >= 0 && (im_col_idx) < im_width) {
            im_data[(im_row_idx + c_im * im_height) * im_width + im_col_idx] +=
                col_data[(c * col_height + h) * col_width + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUDeviceContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUDeviceContext, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUDeviceContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUDeviceContext, double>;

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                    platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& im, const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* col) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col->dims().size() == 5);
    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[3];
    int filter_width = col->dims()[4];
    int col_height = col->dims()[0];
    int col_width = col->dims()[1];

    const T* im_data = im.data<T>();
    T* col_data = col->data<T>();

    for (int col_row_idx = 0; col_row_idx < col_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < col_width; ++col_col_idx) {
        for (int channel = 0; channel < im_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            int im_row_offset =
                col_row_idx * stride[0] + filter_row_idx - padding[0];
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_col_offset =
                  col_col_idx * stride[1] + filter_col_idx - padding[1];

              int col_offset =
                  ((((col_row_idx)*col_width + col_col_idx) * im_channels +
                    channel) *
                       filter_height +
                   filter_row_idx) *
                      filter_width +
                  filter_col_idx;

              int im_offset = (channel * im_height + im_row_offset) * im_width +
                              im_col_offset;
              col_data[col_offset] =
                  (im_row_offset < 0 || im_row_offset >= im_height ||
                   im_col_offset < 0 || im_col_offset >= im_width)
                      ? static_cast<T>(0)
                      : im_data[im_offset];
            }
          }
        }
      }
    }
  }
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                    platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* im) {
    PADDLE_ENFORCE(im->dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int col_height = col.dims()[0];
    int col_width = col.dims()[1];

    PADDLE_ENFORCE_EQ(
        (im_height + padding[0] + padding[2] - filter_height) / stride[0] + 1,
        col_height,
        "Output_height and padding(padding_up, padding_down) are "
        "inconsistent.");
    PADDLE_ENFORCE_EQ(
        (im_width + padding[1] + padding[3] - filter_width) / stride[1] + 1,
        col_width,
        "col_width and padding(padding_left, padding_right) are "
        "inconsistent.");

    T* im_data = im->data<T>();
    const T* col_data = col.data<T>();

    for (int col_row_idx = 0; col_row_idx < col_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < col_width; ++col_col_idx) {
        for (int channel = 0; channel < im_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            int im_row_offset =
                col_row_idx * stride[0] + filter_row_idx - padding[0];
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_col_offset =
                  col_col_idx * stride[1] + filter_col_idx - padding[1];

              int col_offset =
                  (((col_row_idx * col_width + col_col_idx) * im_channels +
                    channel) *
                       filter_height +
                   filter_row_idx) *
                      filter_width +
                  filter_col_idx;

              if (im_row_offset >= 0 && im_row_offset < im_height &&
                  im_col_offset >= 0 && im_col_offset < im_width) {
                int im_offset =
                    (channel * im_height + im_row_offset) * im_width +
                    im_col_offset;
                im_data[im_offset] += col_data[col_offset];
              }
            }
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUDeviceContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUDeviceContext, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUDeviceContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
