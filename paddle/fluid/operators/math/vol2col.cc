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

#include "paddle/fluid/operators/math/vol2col.h"
#include <vector>

namespace paddle {
namespace operators {
namespace math {

/*
 * vol = [input_channels, input_depth, input_height, input_width]
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
template <class T>
class Vol2ColFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& vol,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* col) const {
    PADDLE_ENFORCE(vol.dims().size() == 4);
    PADDLE_ENFORCE(col->dims().size() == 7);

    int input_channels = vol.dims()[0];
    int input_depth = vol.dims()[1];
    int input_height = vol.dims()[2];
    int input_width = vol.dims()[3];
    int filter_depth = col->dims()[1];
    int filter_height = col->dims()[2];
    int filter_width = col->dims()[3];
    int output_depth = col->dims()[4];
    int output_height = col->dims()[5];
    int output_width = col->dims()[6];
    int channels_col =
        input_channels * filter_depth * filter_height * filter_width;

    PADDLE_ENFORCE_EQ((input_depth + 2 * paddings[0] -
                       ((dilations[0] * (filter_depth - 1) + 1))) /
                              strides[0] +
                          1,
                      output_depth,
                      "input_depth and output_depth are "
                      "mismatching.");
    PADDLE_ENFORCE_EQ((input_height + 2 * paddings[1] -
                       ((dilations[1] * (filter_height - 1) + 1))) /
                              strides[1] +
                          1,
                      output_height,
                      "input_height and output_height are "
                      "mismatching.");
    PADDLE_ENFORCE_EQ((input_width + 2 * paddings[2] -
                       ((dilations[2] * (filter_width - 1) + 1))) /
                              strides[2] +
                          1,
                      output_width,
                      "input_width and output_width are "
                      "mismatching.");

    const T* vol_data = vol.data<T>();
    T* col_data = col->data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int d_offset = (c / filter_width / filter_height) % filter_depth;
      int c_in = c / filter_width / filter_height / filter_depth;
      for (int d = 0; d < output_depth; ++d) {
        int d_pad = d * strides[0] - paddings[0] + d_offset * dilations[0];
        for (int h = 0; h < output_height; ++h) {
          int h_pad = h * strides[1] - paddings[1] + h_offset * dilations[1];
          for (int w = 0; w < output_width; ++w) {
            int w_pad = w * strides[2] - paddings[2] + w_offset * dilations[2];

            int col_idx =
                ((c * output_depth + d) * output_height + h) * output_width + w;
            int vol_idx =
                ((c_in * input_depth + d_pad) * input_height + h_pad) *
                    input_width +
                w_pad;
            col_data[col_idx] =
                (h_pad < 0 || h_pad >= input_height || w_pad < 0 ||
                 w_pad >= input_width || d_pad < 0 || d_pad >= input_depth)
                    ? static_cast<T>(0)
                    : vol_data[vol_idx];
          }
        }
      }
    }
  }
};

/*
 * vol = [input_channels,input_depth, input_height, input_width]
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
template <class T>
class Col2VolFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& col,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* vol) const {
    PADDLE_ENFORCE(vol->dims().size() == 4);
    PADDLE_ENFORCE(col.dims().size() == 7);

    int input_channels = vol->dims()[0];
    int input_depth = vol->dims()[1];
    int input_height = vol->dims()[2];
    int input_width = vol->dims()[3];
    int filter_depth = col.dims()[1];
    int filter_height = col.dims()[2];
    int filter_width = col.dims()[3];
    int output_depth = col.dims()[4];
    int output_height = col.dims()[5];
    int output_width = col.dims()[6];
    int channels_col =
        input_channels * filter_depth * filter_height * filter_width;

    PADDLE_ENFORCE_EQ((input_depth + 2 * paddings[0] -
                       ((dilations[0] * (filter_depth - 1) + 1))) /
                              strides[0] +
                          1,
                      output_depth,
                      "input_depth and output_depth are "
                      "mismatching.");
    PADDLE_ENFORCE_EQ((input_height + 2 * paddings[1] -
                       ((dilations[1] * (filter_height - 1) + 1))) /
                              strides[1] +
                          1,
                      output_height,
                      "input_height and output_height are "
                      "mismatching.");
    PADDLE_ENFORCE_EQ((input_width + 2 * paddings[2] -
                       ((dilations[2] * (filter_width - 1) + 1))) /
                              strides[2] +
                          1,
                      output_width,
                      "input_width and output_width are "
                      "mismatching.");
    T* vol_data = vol->data<T>();
    const T* col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int d_offset = (c / filter_width / filter_height) % filter_depth;
      int cIm = c / filter_width / filter_height / filter_depth;
      for (int d = 0; d < output_depth; ++d) {
        int d_pad = d * strides[0] - paddings[0] + d_offset * dilations[0];
        for (int h = 0; h < output_height; ++h) {
          int h_pad = h * strides[1] - paddings[1] + h_offset * dilations[1];
          for (int w = 0; w < output_width; ++w) {
            int w_pad = w * strides[2] - paddings[2] + w_offset * dilations[2];

            if (h_pad >= 0 && h_pad < input_height && w_pad >= 0 &&
                w_pad < input_width && d_pad >= 0 && d_pad < input_depth) {
              int vol_idx =
                  ((cIm * input_depth + d_pad) * input_height + h_pad) *
                      input_width +
                  w_pad;

              int col_idx =
                  ((c * output_depth + d) * output_height + h) * output_width +
                  w;
              vol_data[vol_idx] += col_data[col_idx];
            }
          }
        }
      }
    }
  }
};

template class Vol2ColFunctor<platform::CPUDeviceContext, float>;
template class Vol2ColFunctor<platform::CPUDeviceContext, double>;
template class Col2VolFunctor<platform::CPUDeviceContext, float>;
template class Col2VolFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
