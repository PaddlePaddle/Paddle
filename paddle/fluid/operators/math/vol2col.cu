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

#include <algorithm>
#include <vector>

#include "paddle/fluid/operators/math/vol2col.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void vol2col(int num_kernels,
                        const T* data_vol,
                        int depth,
                        int height,
                        int width,
                        int dilation_d,
                        int dilation_h,
                        int dilation_w,
                        int filter_depth,
                        int filter_height,
                        int filter_width,
                        int stride_depth,
                        int stride_height,
                        int stride_width,
                        int padding_depth,
                        int padding_height,
                        int padding_width,
                        int output_detph,
                        int output_height,
                        int output_width,
                        T* data_col,
                        const DataLayout data_layout) {
  int input_channels =
      num_kernels / output_detph / output_height / output_width;
  int channels_col =
      input_channels * filter_depth * filter_height * filter_width;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
       index += blockDim.x * gridDim.x) {
    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int d_out = (index / output_width / output_height) % output_detph;
    int channel_in = index / output_width / output_height / output_detph;
    int channel_out = channel_in * filter_depth * filter_height * filter_width;
    int w_in = w_out * stride_width - padding_width;
    int h_in = h_out * stride_height - padding_height;
    int d_in = d_out * stride_depth - padding_depth;

    data_col += ((channel_out * output_detph + d_out) * output_height + h_out) *
                    output_width +
                w_out;
    for (int k = 0; k < filter_depth; ++k) {
      for (int i = 0; i < filter_height; ++i) {
        for (int j = 0; j < filter_width; ++j) {
          int d = d_in + k * dilation_d;
          int h = h_in + i * dilation_h;
          int w = w_in + j * dilation_w;
          int vol_idx;
          if (data_layout != DataLayout::kNHWC) {
            vol_idx = ((channel_in * depth + d) * height + h) * width + w;
          } else {
            vol_idx =
                ((d * height + h) * width + w) * input_channels + channel_in;
          }
          *data_col = (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 &&
                       w < width)
                          ? data_vol[vol_idx]
                          : 0;
          data_col += output_detph * output_height * output_width;
        }
      }
    }
  }
}

/*
 * im = [input_channels,intpu_depth, input_height, input_width] for
 * channels_first
 * im = [input_depth, input_height, input_width, input_channels] for
 * channels_last
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
// template <class DeviceContext, class T>
// class Vol2ColFunctor {
//  public:
template <class DeviceContext, class T>
void Vol2ColFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor& vol,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    phi::DenseTensor* col,
    const DataLayout data_layout) const {
  PADDLE_ENFORCE_EQ(vol.dims().size(),
                    4,
                    platform::errors::InvalidArgument(
                        "The dimension of  vol should be 4, but received %d.",
                        vol.dims().size()));
  PADDLE_ENFORCE_EQ(col->dims().size(),
                    7,
                    platform::errors::InvalidArgument(
                        "The dimension of col should be 7, but received %d.",
                        col->dims().size()));

  int input_channels =
      (data_layout != DataLayout::kNHWC ? vol.dims()[0] : vol.dims()[3]);
  int input_depth =
      (data_layout != DataLayout::kNHWC ? vol.dims()[1] : vol.dims()[0]);
  int input_height =
      (data_layout != DataLayout::kNHWC ? vol.dims()[2] : vol.dims()[1]);
  int input_width =
      (data_layout != DataLayout::kNHWC ? vol.dims()[3] : vol.dims()[2]);
  int filter_depth = col->dims()[1];
  int filter_height = col->dims()[2];
  int filter_width = col->dims()[3];
  int output_depth = col->dims()[4];
  int output_height = col->dims()[5];
  int output_width = col->dims()[6];

  bool paddings_size_is_6 = (paddings.size() == 6);
  int pad_d_forth = paddings_size_is_6 ? paddings[0] : paddings[0];
  int pad_d_back = paddings_size_is_6 ? paddings[1] : paddings[0];
  int pad_h_up = paddings_size_is_6 ? paddings[2] : paddings[1];
  int pad_h_down = paddings_size_is_6 ? paddings[3] : paddings[1];
  int pad_w_left = paddings_size_is_6 ? paddings[4] : paddings[2];
  int pad_w_right = paddings_size_is_6 ? paddings[5] : paddings[2];
  auto input_depth_tmp = (input_depth + pad_d_forth + pad_d_back -
                          ((dilations[0] * (filter_depth - 1) + 1))) /
                             strides[0] +
                         1;
  PADDLE_ENFORCE_EQ(input_depth_tmp,
                    output_depth,
                    platform::errors::InvalidArgument(
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
      platform::errors::InvalidArgument(
          "input_height(%d) and output_height(%d) are mismatching.",
          input_height_tmp,
          output_height));
  auto input_width_tmp = (input_width + pad_w_left + pad_w_right -
                          ((dilations[2] * (filter_width - 1) + 1))) /
                             strides[2] +
                         1;
  PADDLE_ENFORCE_EQ(input_width_tmp,
                    output_width,
                    platform::errors::InvalidArgument(
                        "input_width(%d) and output_width(%d) are mismatching.",
                        input_width_tmp,
                        output_width));

  int num_outputs =
      input_channels * output_depth * output_height * output_width;

  int max_threads = 1024;
#ifdef WITH_NV_JETSON
  platform::ChangeThreadNum(context, &max_threads);
#endif

  const int threads = max_threads;
  const int blocks = (num_outputs + max_threads - 1) / max_threads;

  vol2col<T><<<blocks, threads, 0, context.stream()>>>(num_outputs,
                                                       vol.data<T>(),
                                                       input_depth,
                                                       input_height,
                                                       input_width,
                                                       dilations[0],
                                                       dilations[1],
                                                       dilations[2],
                                                       filter_depth,
                                                       filter_height,
                                                       filter_width,
                                                       strides[0],
                                                       strides[1],
                                                       strides[2],
                                                       pad_d_forth,
                                                       pad_h_up,
                                                       pad_w_left,
                                                       output_depth,
                                                       output_height,
                                                       output_width,
                                                       col->data<T>(),
                                                       data_layout);
}
// };

template <class T>
__global__ void col2vol(int num_kernels,
                        const T* data_col,
                        int depth,
                        int height,
                        int width,
                        int dilation_d,
                        int dilation_h,
                        int dilation_w,
                        int filter_depth,
                        int filter_height,
                        int filter_width,
                        int stride_depth,
                        int stride_height,
                        int stride_width,
                        int padding_depth,
                        int padding_height,
                        int padding_width,
                        int output_detph,
                        int output_height,
                        int output_width,
                        T* data_vol,
                        const DataLayout data_layout) {
  const int d_filter_depth = dilation_d * (filter_depth - 1) + 1;
  const int d_filter_height = dilation_h * (filter_height - 1) + 1;
  const int d_filter_width = dilation_w * (filter_width - 1) + 1;

  int input_channels = num_kernels / depth / height / width;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
       index += blockDim.x * gridDim.x) {
    T src_val = 0;
    int w = (data_layout != DataLayout::kNHWC
                 ? index % width + padding_width
                 : (index / input_channels) % width + padding_width);
    int h = (data_layout != DataLayout::kNHWC
                 ? (index / width) % height + padding_height
                 : (index / input_channels / width) % height + padding_height);
    int d = (data_layout != DataLayout::kNHWC
                 ? (index / width / height) % depth + padding_depth
                 : index / input_channels / width / height + padding_depth);
    int c = (data_layout != DataLayout::kNHWC ? index / width / height / depth
                                              : index % input_channels);

    // compute the start and end of the output
    int w_col_start =
        (w < d_filter_width) ? 0 : (w - d_filter_width) / stride_width + 1;
    int w_col_end = min(w / stride_width + 1, output_width);
    int h_col_start =
        (h < d_filter_height) ? 0 : (h - d_filter_height) / stride_height + 1;
    int h_col_end = min(h / stride_height + 1, output_height);
    int d_col_start =
        (d < d_filter_depth) ? 0 : (d - d_filter_depth) / stride_depth + 1;
    int d_col_end = min(d / stride_depth + 1, output_detph);

    for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          int d_off = (d - d_col * stride_depth);
          int h_off = (h - h_col * stride_height);
          int w_off = (w - w_col * stride_width);
          if (d_off % dilation_d == 0 && h_off % dilation_h == 0 &&
              w_off % dilation_w == 0) {
            d_off /= dilation_d;
            h_off /= dilation_h;
            w_off /= dilation_w;

            int data_col_index =
                (((((c * filter_depth + d_off) * filter_height + h_off) *
                       filter_width +
                   w_off)));
            data_col_index =
                ((data_col_index * output_detph + d_col) * output_height +
                 h_col) *
                    output_width +
                w_col;
            src_val += data_col[data_col_index];
          }
        }
      }
    }
    data_vol[index] = src_val;
  }
}

/*
 * im = [input_channels,intpu_depth, input_height, input_width] for
 * channels_first
 * im = [input_depth, input_height, input_width, input_channels] for
 * channels_last
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
// template <class DeviceContext, class T>
// class Col2VolFunctor<DeviceContext, T> {
//  public:
template <class DeviceContext, class T>
void Col2VolFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor& col,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    phi::DenseTensor* vol,
    const DataLayout data_layout) const {
  PADDLE_ENFORCE_EQ(vol->dims().size(),
                    4,
                    platform::errors::InvalidArgument(
                        "The dimension of vol  should be 4, but received %d.",
                        vol->dims().size()));
  PADDLE_ENFORCE_EQ(col.dims().size(),
                    7,
                    platform::errors::InvalidArgument(
                        "The dimension of col  should be 7, but received %d.",
                        col.dims().size()));

  int input_channels =
      (data_layout != DataLayout::kNHWC ? vol->dims()[0] : vol->dims()[3]);
  int input_depth =
      (data_layout != DataLayout::kNHWC ? vol->dims()[1] : vol->dims()[0]);
  int input_height =
      (data_layout != DataLayout::kNHWC ? vol->dims()[2] : vol->dims()[1]);
  int input_width =
      (data_layout != DataLayout::kNHWC ? vol->dims()[3] : vol->dims()[2]);
  int filter_depth = col.dims()[1];
  int filter_height = col.dims()[2];
  int filter_width = col.dims()[3];
  int output_depth = col.dims()[4];
  int output_height = col.dims()[5];
  int output_width = col.dims()[6];

  bool paddings_size_is_6 = (paddings.size() == 6);
  int pad_d_forth = paddings_size_is_6 ? paddings[0] : paddings[0];
  int pad_d_back = paddings_size_is_6 ? paddings[1] : paddings[0];
  int pad_h_up = paddings_size_is_6 ? paddings[2] : paddings[1];
  int pad_h_down = paddings_size_is_6 ? paddings[3] : paddings[1];
  int pad_w_left = paddings_size_is_6 ? paddings[4] : paddings[2];
  int pad_w_right = paddings_size_is_6 ? paddings[5] : paddings[2];

  auto input_depth_tmp = (input_depth + pad_d_forth + pad_d_back -
                          ((dilations[0] * (filter_depth - 1) + 1))) /
                             strides[0] +
                         1;
  PADDLE_ENFORCE_EQ(input_depth_tmp,
                    output_depth,
                    platform::errors::InvalidArgument(
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
      platform::errors::InvalidArgument(
          "input_height(%d) and output_height(%d) are mismatching.",
          input_height_tmp,
          output_height));
  auto input_width_tmp = (input_width + pad_w_left + pad_w_right -
                          ((dilations[2] * (filter_width - 1) + 1))) /
                             strides[2] +
                         1;
  PADDLE_ENFORCE_EQ(input_width_tmp,
                    output_width,
                    platform::errors::InvalidArgument(
                        "input_width(%d) and output_width(%d) are mismatching.",
                        input_width_tmp,
                        output_width));

  int num_kernels = input_channels * input_depth * input_height * input_width;

  int max_threads = 1024;
#ifdef WITH_NV_JETSON
  platform::ChangeThreadNum(context, &max_threads);
#endif

  const int threads = max_threads;
  const int blocks = (num_kernels + max_threads - 1) / max_threads;

  col2vol<T><<<blocks, threads, 0, context.stream()>>>(num_kernels,
                                                       col.data<T>(),
                                                       input_depth,
                                                       input_height,
                                                       input_width,
                                                       dilations[0],
                                                       dilations[1],
                                                       dilations[2],
                                                       filter_depth,
                                                       filter_height,
                                                       filter_width,
                                                       strides[0],
                                                       strides[1],
                                                       strides[2],
                                                       pad_d_forth,
                                                       pad_h_up,
                                                       pad_w_left,
                                                       output_depth,
                                                       output_height,
                                                       output_width,
                                                       vol->data<T>(),
                                                       data_layout);
}
// };

template class Vol2ColFunctor<phi::GPUContext, float>;
template class Vol2ColFunctor<phi::GPUContext, double>;

template class Col2VolFunctor<phi::GPUContext, float>;
template class Col2VolFunctor<phi::GPUContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
