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

#include "paddle/phi/kernels/funcs/vol2col.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"

namespace phi::funcs {

/*
 * vol = [input_channels, input_depth, input_height, input_width]
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
template <class T>
class Vol2ColFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& vol,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  phi::DenseTensor* col,
                  const DataLayout data_layout) const {
    PADDLE_ENFORCE_EQ(vol.dims().size(),
                      4,
                      common::errors::InvalidArgument(
                          "The dimension of vol should be 4, but received %d.",
                          vol.dims().size()));

    PADDLE_ENFORCE_EQ(col->dims().size(),
                      7,
                      common::errors::InvalidArgument(
                          "The dimension of col should be 7, but received %d.",
                          col->dims().size()));

    int input_channels = static_cast<int>(
        data_layout != DataLayout::kNHWC ? vol.dims()[0] : vol.dims()[3]);
    int64_t input_depth =
        (data_layout != DataLayout::kNHWC ? vol.dims()[1] : vol.dims()[0]);
    int64_t input_height =
        (data_layout != DataLayout::kNHWC ? vol.dims()[2] : vol.dims()[1]);
    int64_t input_width =
        (data_layout != DataLayout::kNHWC ? vol.dims()[3] : vol.dims()[2]);
    int filter_depth = static_cast<int>(col->dims()[1]);
    int filter_height = static_cast<int>(col->dims()[2]);
    int filter_width = static_cast<int>(col->dims()[3]);
    int64_t output_depth = col->dims()[4];
    int64_t output_height = col->dims()[5];
    int64_t output_width = col->dims()[6];
    int channels_col =
        input_channels * filter_depth * filter_height * filter_width;

    // changed
    bool paddings_size_is_6 = (paddings.size() == 6);
    int pad_d_forth = paddings[0];
    int pad_d_back = paddings_size_is_6 ? paddings[1] : paddings[0];
    int pad_h_up = paddings_size_is_6 ? paddings[2] : paddings[1];
    int pad_h_down = paddings_size_is_6 ? paddings[3] : paddings[1];
    int pad_w_left = paddings_size_is_6 ? paddings[4] : paddings[2];
    int pad_w_right = paddings_size_is_6 ? paddings[5] : paddings[2];

    auto input_depth_tmp = (input_depth + pad_d_forth + pad_d_back -
                            ((dilations[0] * (filter_depth - 1) + 1))) /
                               strides[0] +
                           1;
    PADDLE_ENFORCE_EQ(
        input_depth_tmp,
        output_depth,
        common::errors::InvalidArgument(
            "input_depth(%d) and output_depth(%d) are mismatching.",
            input_depth_tmp,
            output_depth));
    auto input_height_tmp = (input_height + pad_h_up + pad_h_down -
                             ((dilations[1] * (filter_height - 1) + 1))) /
                                strides[1] +
                            1;
    PADDLE_ENFORCE_EQ(
        input_height_tmp,
        output_height,
        common::errors::InvalidArgument(
            "input_height(%d) and output_height(%d) are mismatching.",
            input_height_tmp,
            output_height));
    auto input_width_tmp = (input_width + pad_w_left + pad_w_right -
                            ((dilations[2] * (filter_width - 1) + 1))) /
                               strides[2] +
                           1;
    PADDLE_ENFORCE_EQ(
        input_width_tmp,
        output_width,
        common::errors::InvalidArgument(
            "input_width(%d) and output_width(%d) are mismatching.",
            input_width_tmp,
            output_width));
    const T* vol_data = vol.data<T>();
    T* col_data = col->data<T>();
    for (auto c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int d_offset = (c / filter_width / filter_height) % filter_depth;
      int64_t c_in = c / filter_width / filter_height / filter_depth;
      for (auto d = 0; d < output_depth; ++d) {
        int64_t d_pad = d * strides[0] - pad_d_forth + d_offset * dilations[0];
        for (auto h = 0; h < output_height; ++h) {
          int64_t h_pad = h * strides[1] - pad_h_up + h_offset * dilations[1];
          for (auto w = 0; w < output_width; ++w) {
            int64_t w_pad =
                w * strides[2] - pad_w_left + w_offset * dilations[2];

            int64_t col_idx =
                ((c * output_depth + d) * output_height + h) * output_width + w;
            int64_t vol_idx = 0;
            if (data_layout != DataLayout::kNHWC) {
              vol_idx = ((c_in * input_depth + d_pad) * input_height + h_pad) *
                            input_width +
                        w_pad;
            } else {
              vol_idx = ((d_pad * input_height + h_pad) * input_width + w_pad) *
                            input_channels +
                        c_in;
            }
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
class Col2VolFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& col,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  phi::DenseTensor* vol,
                  const DataLayout data_layout) const {
    PADDLE_ENFORCE_EQ(vol->dims().size(),
                      4,
                      common::errors::InvalidArgument(
                          "The dimension of vol should be 4, but received %d.",
                          vol->dims().size()));

    PADDLE_ENFORCE_EQ(col.dims().size(),
                      7,
                      common::errors::InvalidArgument(
                          "The dimension of col  should be 7, but received %d.",
                          col.dims().size()));

    int input_channels = static_cast<int>(
        data_layout != DataLayout::kNHWC ? vol->dims()[0] : vol->dims()[3]);
    int input_depth = static_cast<int>(
        data_layout != DataLayout::kNHWC ? vol->dims()[1] : vol->dims()[0]);
    int input_height = static_cast<int>(
        data_layout != DataLayout::kNHWC ? vol->dims()[2] : vol->dims()[1]);
    int input_width = static_cast<int>(
        data_layout != DataLayout::kNHWC ? vol->dims()[3] : vol->dims()[2]);
    int filter_depth = static_cast<int>(col.dims()[1]);
    int filter_height = static_cast<int>(col.dims()[2]);
    int filter_width = static_cast<int>(col.dims()[3]);
    int output_depth = static_cast<int>(col.dims()[4]);
    int output_height = static_cast<int>(col.dims()[5]);
    int output_width = static_cast<int>(col.dims()[6]);
    int channels_col =
        input_channels * filter_depth * filter_height * filter_width;

    bool paddings_size_is_6 = (paddings.size() == 6);
    int pad_d_forth = paddings[0];
    int pad_d_back = paddings_size_is_6 ? paddings[1] : paddings[0];
    int pad_h_up = paddings_size_is_6 ? paddings[2] : paddings[1];
    int pad_h_down = paddings_size_is_6 ? paddings[3] : paddings[1];
    int pad_w_left = paddings_size_is_6 ? paddings[4] : paddings[2];
    int pad_w_right = paddings_size_is_6 ? paddings[5] : paddings[2];

    auto input_depth_tmp = (input_depth + pad_d_forth + pad_d_back -
                            ((dilations[0] * (filter_depth - 1) + 1))) /
                               strides[0] +
                           1;
    PADDLE_ENFORCE_EQ(
        input_depth_tmp,
        output_depth,
        common::errors::InvalidArgument(
            "input_depth(%d) and output_depth(%d) are mismatching.",
            input_depth_tmp,
            output_depth));
    auto input_height_tmp = (input_height + pad_h_up + pad_h_down -
                             ((dilations[1] * (filter_height - 1) + 1))) /
                                strides[1] +
                            1;
    PADDLE_ENFORCE_EQ(
        input_height_tmp,
        output_height,
        common::errors::InvalidArgument(
            "input_height(%d) and output_height(%d) are mismatching.",
            input_height_tmp,
            output_height));
    auto input_width_tmp = (input_width + pad_w_left + pad_w_right -
                            ((dilations[2] * (filter_width - 1) + 1))) /
                               strides[2] +
                           1;
    PADDLE_ENFORCE_EQ(
        input_width_tmp,
        output_width,
        common::errors::InvalidArgument(
            "input_width(%d) and output_width(%d) are mismatching.",
            input_width_tmp,
            output_width));
    T* vol_data = vol->data<T>();
    const T* col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int d_offset = (c / filter_width / filter_height) % filter_depth;
      int cIm = c / filter_width / filter_height / filter_depth;
      for (int d = 0; d < output_depth; ++d) {
        int d_pad = d * strides[0] - pad_d_forth + d_offset * dilations[0];
        for (int h = 0; h < output_height; ++h) {
          int h_pad = h * strides[1] - pad_h_up + h_offset * dilations[1];
          for (int w = 0; w < output_width; ++w) {
            int w_pad = w * strides[2] - pad_w_left + w_offset * dilations[2];

            if (h_pad >= 0 && h_pad < input_height && w_pad >= 0 &&
                w_pad < input_width && d_pad >= 0 && d_pad < input_depth) {
              int vol_idx = 0;
              if (data_layout != DataLayout::kNHWC) {
                vol_idx = ((cIm * input_depth + d_pad) * input_height + h_pad) *
                              input_width +
                          w_pad;
              } else {
                vol_idx =
                    ((d_pad * input_height + h_pad) * input_width + w_pad) *
                        input_channels +
                    cIm;
              }
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

template class Vol2ColFunctor<phi::CPUContext, float>;
template class Vol2ColFunctor<phi::CPUContext, double>;

template class Col2VolFunctor<phi::CPUContext, float>;
template class Col2VolFunctor<phi::CPUContext, double>;

}  // namespace phi::funcs
