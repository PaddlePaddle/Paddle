/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/im2col.h"

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
                    platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& im, framework::Tensor& col,
                  int stride_height, int stride_width, int padding_up,
                  int padding_down, int padding_left, int padding_right) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);

    int input_channels = im.dims()[0];
    int input_height = im.dims()[1];
    int input_width = im.dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int output_height = col.dims()[3];
    int output_width = col.dims()[4];

    PADDLE_ENFORCE_EQ(
        (input_height + padding_up + padding_down - filter_height) /
                stride_height +
            1,
        output_height,
        "Output_height and padding(padding_up, padding_down) are "
        "inconsistent.");
    PADDLE_ENFORCE_EQ(
        (input_width + padding_left + padding_right - filter_width) /
                stride_width +
            1,
        output_width,
        "output_width and padding(padding_left, padding_right) are "
        "inconsistent.");

    int channels_col = input_channels * filter_height * filter_width;

    const T* im_data = im.data<T>();
    T* col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int c_im = c / filter_width / filter_height;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          int im_row_idx = h * stride_height + h_offset - padding_up;
          int im_col_idx = w * stride_width + w_offset - padding_left;

          if (im_row_idx < 0 || im_row_idx >= input_height || im_col_idx < 0 ||
              im_col_idx >= input_width) {
            col_data[(c * output_height + h) * output_width + w] = T(0);
          } else {
            im_row_idx += c_im * input_height;
            col_data[(c * output_height + h) * output_width + w] =
                im_data[im_row_idx * input_width + im_col_idx];
          }
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
                    platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context, framework::Tensor& im,
                  const framework::Tensor& col, int stride_height,
                  int stride_width, int padding_up, int padding_down,
                  int padding_left, int padding_right) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);
    int input_channels = im.dims()[0];
    int input_height = im.dims()[1];
    int input_width = im.dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int output_height = col.dims()[3];
    int output_width = col.dims()[4];

    PADDLE_ENFORCE_EQ(
        (input_height + padding_up + padding_down - filter_height) /
                stride_height +
            1,
        output_height,
        "Output_height and padding(padding_up, padding_down) are "
        "inconsistent.");
    PADDLE_ENFORCE_EQ(
        (input_width + padding_left + padding_right - filter_width) /
                stride_width +
            1,
        output_width,
        "output_width and padding(padding_left, padding_right) are "
        "inconsistent.");

    int channels_col = input_channels * filter_height * filter_width;

    T* im_data = im.data<T>();
    const T* col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int c_im = c / filter_width / filter_height;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          int im_row_idx = h * stride_height + h_offset - padding_up;
          int im_col_idx = w * stride_width + w_offset - padding_left;

          if ((im_row_idx) >= 0 && (im_row_idx) < input_height &&
              (im_col_idx) >= 0 && (im_col_idx) < input_width) {
            im_row_idx += c_im * input_height;
            im_data[im_row_idx * input_width + im_col_idx] +=
                col_data[(c * output_height + h) * output_width + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUPlace, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUPlace, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUPlace, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CPUPlace, double>;

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                    platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& im, framework::Tensor& col,
                  int stride_height, int stride_width, int padding_up,
                  int padding_down, int padding_left, int padding_right) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);
    int input_channels = im.dims()[0];
    int input_height = im.dims()[1];
    int input_width = im.dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int output_height = col.dims()[0];
    int output_width = col.dims()[1];

    PADDLE_ENFORCE_EQ(
        (input_height + padding_up + padding_down - filter_height) /
                stride_height +
            1,
        output_height,
        "Output_height and padding(padding_up, padding_down) are "
        "inconsistent.");
    PADDLE_ENFORCE_EQ(
        (input_width + padding_left + padding_right - filter_width) /
                stride_width +
            1,
        output_width,
        "output_width and padding(padding_left, padding_right) are "
        "inconsistent.");

    const T* im_data = im.data<T>();
    T* col_data = col.data<T>();

    for (int col_row_idx = 0; col_row_idx < output_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < output_width; ++col_col_idx) {
        for (int channel = 0; channel < input_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_row_offset =
                  col_row_idx * stride_height + filter_row_idx - padding_up;
              int im_col_offset =
                  col_col_idx * stride_width + filter_col_idx - padding_left;
              int col_offset = ((((col_row_idx)*output_width + col_col_idx) *
                                     input_channels +
                                 channel) *
                                    filter_height +
                                filter_row_idx) *
                                   filter_width +
                               filter_col_idx;
              if (im_row_offset < 0 || im_row_offset >= input_height ||
                  im_col_offset < 0 || im_col_offset >= input_width) {
                col_data[col_offset] = T(0);
              } else {
                int im_offset =
                    (channel * input_height + im_row_offset) * input_width +
                    im_col_offset;
                col_data[col_offset] = im_data[im_offset];
              }
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
                    platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context, framework::Tensor& im,
                  const framework::Tensor& col, int stride_height,
                  int stride_width, int padding_up, int padding_down,
                  int padding_left, int padding_right) {
    PADDLE_ENFORCE(im.dims().size() == 3);
    PADDLE_ENFORCE(col.dims().size() == 5);
    int input_channels = im.dims()[0];
    int input_height = im.dims()[1];
    int input_width = im.dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int output_height = col.dims()[0];
    int output_width = col.dims()[1];

    PADDLE_ENFORCE_EQ(
        (input_height + padding_up + padding_down - filter_height) /
                stride_height +
            1,
        output_height,
        "Output_height and padding(padding_up, padding_down) are "
        "inconsistent.");
    PADDLE_ENFORCE_EQ(
        (input_width + padding_left + padding_right - filter_width) /
                stride_width +
            1,
        output_width,
        "output_width and padding(padding_left, padding_right) are "
        "inconsistent.");

    T* im_data = im.data<T>();
    const T* col_data = col.data<T>();

    for (int col_row_idx = 0; col_row_idx < output_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < output_width; ++col_col_idx) {
        for (int channel = 0; channel < input_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_row_offset =
                  col_row_idx * stride_height + filter_row_idx - padding_up;
              int im_col_offset =
                  col_col_idx * stride_width + filter_col_idx - padding_left;
              int col_offset = (((col_row_idx * output_width + col_col_idx) *
                                     input_channels +
                                 channel) *
                                    filter_height +
                                filter_row_idx) *
                                   filter_width +
                               filter_col_idx;
              if (im_row_offset >= 0 && im_row_offset < input_height &&
                  im_col_offset >= 0 && im_col_offset < input_width) {
                int im_offset =
                    (channel * input_height + im_row_offset) * input_width +
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
                             platform::CPUPlace, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUPlace, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUPlace, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
