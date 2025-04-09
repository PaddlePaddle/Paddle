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

#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/im2col_cfo_cpu.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace phi::funcs {

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T, typename DeviceContext>
class Im2ColFunctor<phi::funcs::ColFormat::kCFO, DeviceContext, T> {
 public:
  void operator()(const DeviceContext& context UNUSED,
                  const phi::DenseTensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* col,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im.dims().size(),
                      3,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im.dims()));
    PADDLE_ENFORCE_EQ(col->dims().size(),
                      5,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col->dims()));

    if (stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
        dilation[1] == 1) {
      if (padding[0] == 0 && padding[1] == 0 && padding[2] == 0 &&
          padding[3] == 0) {
        im2col_sh1sw1dh1dw1ph0pw0<T>(im, col, data_layout);
        return;
      } else if (padding[0] == 1 && padding[1] == 1 && padding[2] == 1 &&
                 padding[3] == 1) {
        im2col_sh1sw1dh1dw1ph1pw1<T>(im, col, data_layout);
        return;
      }
      // TODO(TJ): complete padding >=2
    }
    im2col_common<T>(im, dilation, stride, padding, col, data_layout);
  }
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T, typename DeviceContext>
class Col2ImFunctor<phi::funcs::ColFormat::kCFO, DeviceContext, T> {
 public:
  void operator()(const DeviceContext& context UNUSED,
                  const phi::DenseTensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* im,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im->dims().size(),
                      3,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im->dims()));
    PADDLE_ENFORCE_EQ(col.dims().size(),
                      5,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col.dims()));
    int im_channels = static_cast<int>(
        data_layout != DataLayout::kNHWC ? im->dims()[0] : im->dims()[2]);
    int im_height = static_cast<int>(
        data_layout != DataLayout::kNHWC ? im->dims()[1] : im->dims()[0]);
    int im_width = static_cast<int>(
        data_layout != DataLayout::kNHWC ? im->dims()[2] : im->dims()[1]);
    int filter_height = static_cast<int>(col.dims()[1]);
    int filter_width = static_cast<int>(col.dims()[2]);
    int col_height = static_cast<int>(col.dims()[3]);
    int col_width = static_cast<int>(col.dims()[4]);

    PADDLE_ENFORCE_EQ(
        (im_height + padding[0] + padding[2] -
         ((dilation[0] * (filter_height - 1) + 1))) /
                stride[0] +
            1,
        col_height,
        common::errors::InvalidArgument("Output_height and padding(padding_up, "
                                        "padding_down) are inconsistent."));
    PADDLE_ENFORCE_EQ(
        (im_width + padding[1] + padding[3] -
         ((dilation[1] * (filter_width - 1) + 1))) /
                stride[1] +
            1,
        col_width,
        common::errors::InvalidArgument("Output_height and padding(padding_up, "
                                        "padding_down) are inconsistent."));

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
            int im_offset = 0;
            if (data_layout != DataLayout::kNHWC) {
              im_offset =
                  (c_im * im_height + im_row_idx) * im_width + im_col_idx;
            } else {
              im_offset =
                  (im_row_idx * im_width + im_col_idx) * im_channels + c_im;
            }
            im_data[im_offset] +=
                col_data[(c * col_height + h) * col_width + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             float>;
template class Im2ColFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             double>;
template class Im2ColFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             phi::dtype::complex<float>>;
template class Im2ColFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             phi::dtype::complex<double>>;
template class Col2ImFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             float>;
template class Col2ImFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             double>;
template class Col2ImFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             phi::dtype::complex<float>>;
template class Col2ImFunctor<phi::funcs::ColFormat::kCFO,
                             phi::CPUContext,
                             phi::dtype::complex<double>>;

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T, typename DeviceContext>
class Im2ColFunctor<phi::funcs::ColFormat::kOCF, DeviceContext, T> {
 public:
  void operator()(const DeviceContext& context UNUSED,
                  const phi::DenseTensor& im,
                  const std::vector<int>& dilation UNUSED,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* col,
                  const DataLayout data_layout UNUSED) {
    PADDLE_ENFORCE_EQ(im.dims().size(),
                      3,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im.dims()));
    PADDLE_ENFORCE_EQ(col->dims().size(),
                      5,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col->dims()));
    int im_channels = static_cast<int>(im.dims()[0]);
    int im_height = static_cast<int>(im.dims()[1]);
    int im_width = static_cast<int>(im.dims()[2]);
    int filter_height = static_cast<int>(col->dims()[3]);
    int filter_width = static_cast<int>(col->dims()[4]);
    int col_height = static_cast<int>(col->dims()[0]);
    int col_width = static_cast<int>(col->dims()[1]);

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
template <class T, typename DeviceContext>
class Col2ImFunctor<phi::funcs::ColFormat::kOCF, DeviceContext, T> {
 public:
  void operator()(const DeviceContext& context UNUSED,
                  const phi::DenseTensor& col,
                  const std::vector<int>& dilation UNUSED,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* im,
                  const DataLayout data_layout UNUSED) {
    PADDLE_ENFORCE_EQ(im->dims().size(),
                      3,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im->dims()));
    PADDLE_ENFORCE_EQ(col.dims().size(),
                      5,
                      common::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col.dims()));
    int im_channels = static_cast<int>(im->dims()[0]);
    int im_height = static_cast<int>(im->dims()[1]);
    int im_width = static_cast<int>(im->dims()[2]);
    int filter_height = static_cast<int>(col.dims()[3]);
    int filter_width = static_cast<int>(col.dims()[4]);
    int col_height = static_cast<int>(col.dims()[0]);
    int col_width = static_cast<int>(col.dims()[1]);

    PADDLE_ENFORCE_EQ(
        (im_height + padding[0] + padding[2] - filter_height) / stride[0] + 1,
        col_height,
        common::errors::InvalidArgument(
            "Output_height and padding(padding_up, padding_down) "
            "are inconsistent."));
    PADDLE_ENFORCE_EQ(
        (im_width + padding[1] + padding[3] - filter_width) / stride[1] + 1,
        col_width,
        common::errors::InvalidArgument("col_width and padding(padding_left, "
                                        "padding_right) are inconsistent."));

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

template class Im2ColFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             float>;
template class Im2ColFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             double>;
template class Im2ColFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             phi::dtype::complex<float>>;
template class Im2ColFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             phi::dtype::complex<double>>;
template class Col2ImFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             float>;
template class Col2ImFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             double>;
template class Col2ImFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             phi::dtype::complex<float>>;
template class Col2ImFunctor<phi::funcs::ColFormat::kOCF,
                             phi::CPUContext,
                             phi::dtype::complex<double>>;
}  // namespace phi::funcs
