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

#include "hip/hip_runtime.h"
#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/math/vol2col.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void vol2col(int num_kernels, const T* data_vol, int depth,
                        int height, int width, int dilation_d, int dilation_h,
                        int dilation_w, int filter_depth, int filter_height,
                        int filter_width, int stride_depth, int stride_height,
                        int stride_width, int padding_depth, int padding_height,
                        int padding_width, int output_detph, int output_height,
                        int output_width, T* data_col) {
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
    data_vol += ((channel_in * depth + d_in) * height + h_in) * width + w_in;
    for (int k = 0; k < filter_depth; ++k) {
      for (int i = 0; i < filter_height; ++i) {
        for (int j = 0; j < filter_width; ++j) {
          int d = d_in + k * dilation_d;
          int h = h_in + i * dilation_h;
          int w = w_in + j * dilation_w;
          int col_idx = (k * dilation_d * height + i * dilation_h) * width +
                        j * dilation_w;
          *data_col = (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 &&
                       w < width)
                          ? data_vol[col_idx]
                          : 0;
          data_col += output_detph * output_height * output_width;
        }
      }
    }
  }
}

/*
 * im = [input_channels,intpu_depth, input_height, input_width]
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
template <class T>
class Vol2ColFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
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

    PADDLE_ENFORCE_EQ((input_depth + 2 * paddings[0] -
                       ((dilations[0] * (filter_depth - 1) + 1))) /
                              strides[0] +
                          1,
                      output_depth,
                      "input_depth and output_depth are "
                      "Mismatching.");
    PADDLE_ENFORCE_EQ((input_height + 2 * paddings[1] -
                       ((dilations[1] * (filter_height - 1) + 1))) /
                              strides[1] +
                          1,
                      output_height,
                      "input_height and output_height are "
                      "Mismatching.");
    PADDLE_ENFORCE_EQ((input_width + 2 * paddings[2] -
                       ((dilations[2] * (filter_width - 1) + 1))) /
                              strides[2] +
                          1,
                      output_width,
                      "input_width and output_width are "
                      "Mismatching.");

    int num_outputs =
        input_channels * output_depth * output_height * output_width;

    const int threads = 1024;
    const int blocks = (num_outputs + 1024 - 1) / 1024;
    hipLaunchKernelGGL((vol2col<T>), dim3(blocks), dim3(threads), 0,
                     context.stream(),
        num_outputs, vol.data<T>(), input_depth, input_height, input_width,
        dilations[0], dilations[1], dilations[2], filter_depth, filter_height,
        filter_width, strides[0], strides[1], strides[2], paddings[0],
        paddings[1], paddings[2], output_depth, output_height, output_width,
        col->data<T>());
  }
};

template <class T>
__global__ void col2vol(int num_kernels, const T* data_col, int depth,
                        int height, int width, int dilation_d, int dilation_h,
                        int dilation_w, int filter_depth, int filter_height,
                        int filter_width, int stride_depth, int stride_height,
                        int stride_width, int padding_depth, int padding_height,
                        int padding_width, int output_detph, int output_height,
                        int output_width, T* data_vol) {
  const int d_filter_depth = dilation_d * (filter_depth - 1) + 1;
  const int d_filter_height = dilation_h * (filter_height - 1) + 1;
  const int d_filter_width = dilation_w * (filter_width - 1) + 1;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
       index += blockDim.x * gridDim.x) {
    T src_val = 0;
    int w = index % width + padding_width;
    int h = (index / width) % height + padding_height;
    int d = (index / width / height) % depth + padding_depth;
    int c = index / width / height / depth;

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
 * im = [input_channels, input_depth, input_height, input_width]
 * col =
 *   [input_channels, filter_depth, filter_height, filter_width,
 *                    output_depth, output_height, output_width]
 */
template <class T>
class Col2VolFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
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

    PADDLE_ENFORCE_EQ((input_depth + 2 * paddings[0] -
                       ((dilations[0] * (filter_depth - 1) + 1))) /
                              strides[0] +
                          1,
                      output_depth,
                      "input_depth and output_depth are "
                      "Mismatching.");
    PADDLE_ENFORCE_EQ((input_height + 2 * paddings[1] -
                       ((dilations[1] * (filter_height - 1) + 1))) /
                              strides[1] +
                          1,
                      output_height,
                      "input_height and output_height are "
                      "Mismatching.");
    PADDLE_ENFORCE_EQ((input_width + 2 * paddings[2] -
                       ((dilations[2] * (filter_width - 1) + 1))) /
                              strides[2] +
                          1,
                      output_width,
                      "input_width and output_width are "
                      "Mismatching.");

    int num_kernels = input_channels * input_depth * input_height * input_width;

    const int threads = 1024;
    const int blocks = (num_kernels + 1024 - 1) / 1024;

    hipLaunchKernelGGL((col2vol<T>), dim3(blocks), dim3(threads), 0,
                     context.stream(),
        num_kernels, col.data<T>(), input_depth, input_height, input_width,
        dilations[0], dilations[1], dilations[2], filter_depth, filter_height,
        filter_width, strides[0], strides[1], strides[2], paddings[0],
        paddings[1], paddings[2], output_depth, output_height, output_width,
        vol->data<T>());
  }
};

template class Vol2ColFunctor<platform::CUDADeviceContext, float>;
template class Vol2ColFunctor<platform::CUDADeviceContext, double>;
template class Col2VolFunctor<platform::CUDADeviceContext, float>;
template class Col2VolFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
