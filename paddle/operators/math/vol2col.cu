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

#include "paddle/operators/math/vol2col.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void vol2col(int num_kernels, const T* data_vol, int depth,
                        int height, int width, int filter_depth,
                        int filter_height, int filter_width, int stride_depth,
                        int stride_height, int stride_width, int padding_depth,
                        int padding_height, int padding_width, int output_detph,
                        int output_height, int output_width, T* data_col) {
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
          int d = d_in + k;
          int h = h_in + i;
          int w = w_in + j;
          *data_col = (d >= 0 && d < depth && h >= 0 && h < height && w >= 0 &&
                       w < width)
                          ? data_vol[(k * height + i) * width + j]
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
class Vol2ColFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& vol, framework::Tensor& col,
                  int stride_depth, int stride_height, int stride_width,
                  int padding_depth, int padding_height,
                  int padding_width) const {
    PADDLE_ENFORCE(vol.dims().size() == 4);
    PADDLE_ENFORCE(col.dims().size() == 7);

    int input_channels = vol.dims()[0];
    int input_depth = vol.dims()[1];
    int input_height = vol.dims()[2];
    int input_width = vol.dims()[3];
    int filter_depth = col.dims()[1];
    int filter_height = col.dims()[2];
    int filter_width = col.dims()[3];
    int output_depth = col.dims()[4];
    int output_height = col.dims()[5];
    int output_width = col.dims()[6];

    int num_outputs =
        input_channels * output_depth * output_height * output_width;

    const int threads = 1024;
    const int blocks = (num_outputs + 1024 - 1) / 1024;
    vol2col<T><<<blocks, threads, 0,
                 reinterpret_cast<const platform::CUDADeviceContext&>(context)
                     .stream()>>>(
        num_outputs, vol.data<T>(), input_depth, input_height, input_width,
        filter_depth, filter_height, filter_width, stride_depth, stride_height,
        stride_width, padding_depth, padding_height, padding_width,
        output_depth, output_height, output_width, col.data<T>());
  }
};

template <class T>
__global__ void col2vol(int num_kernels, const T* data_col, int depth,
                        int height, int width, int filter_depth,
                        int filter_height, int filter_width, int stride_depth,
                        int stride_height, int stride_width, int padding_depth,
                        int padding_height, int padding_width, int output_detph,
                        int output_height, int output_width, T* data_vol) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
       index += blockDim.x * gridDim.x) {
    T src_val = 0;
    int w = index % width + padding_width;
    int h = (index / width) % height + padding_height;
    int d = (index / width / height) % depth + padding_depth;
    int c = index / width / height / depth;
    // compute the start and end of the output
    int w_col_start =
        (w < filter_width) ? 0 : (w - filter_width) / stride_width + 1;
    int w_col_end = min(w / stride_width + 1, output_width);
    int h_col_start =
        (h < filter_height) ? 0 : (h - filter_height) / stride_height + 1;
    int h_col_end = min(h / stride_height + 1, output_height);
    int d_col_start =
        (d < filter_depth) ? 0 : (d - filter_depth) / stride_depth + 1;
    int d_col_end = min(d / stride_depth + 1, output_detph);

    int offset = (c * filter_depth * filter_height * filter_width +
                  d * filter_width * filter_height + h * filter_width + w) *
                 output_detph * output_height * output_width;

    int coeff_d_col =
        (1 - stride_depth * filter_width * filter_height * output_detph) *
        output_height * output_width;
    int coeff_h_col =
        (1 - stride_height * filter_width * output_detph * output_height) *
        output_width;
    int coeff_w_col =
        (1 - stride_width * output_detph * output_height * output_width);

    for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          src_val += data_col[offset + d_col * coeff_d_col +
                              h_col * coeff_h_col + w_col * coeff_w_col];
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
class Col2VolFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& vol, const framework::Tensor& col,
                  int stride_depth, int stride_height, int stride_width,
                  int padding_depth, int padding_height,
                  int padding_width) const {
    PADDLE_ENFORCE(vol.dims().size() == 4);
    PADDLE_ENFORCE(col.dims().size() == 7);

    int input_channels = vol.dims()[0];
    int input_depth = vol.dims()[1];
    int input_height = vol.dims()[2];
    int input_width = vol.dims()[3];
    int filter_depth = col.dims()[1];
    int filter_height = col.dims()[2];
    int filter_width = col.dims()[3];
    int output_depth = col.dims()[4];
    int output_height = col.dims()[5];
    int output_width = col.dims()[6];

    int num_kernels = input_channels * input_depth * input_height * input_width;

    const int threads = 1024;
    const int blocks = (num_kernels + 1024 - 1) / 1024;

    col2vol<T><<<blocks, threads, 0,
                 reinterpret_cast<const platform::CUDADeviceContext&>(context)
                     .stream()>>>(
        num_kernels, col.data<T>(), input_depth, input_height, input_width,
        filter_depth, filter_height, filter_width, stride_depth, stride_height,
        stride_width, padding_depth, padding_height, padding_width,
        output_depth, output_height, output_width, vol.data<T>());
  }
};

template class Vol2ColFunctor<platform::GPUPlace, float>;
template class Vol2ColFunctor<platform::GPUPlace, double>;
template class Col2VolFunctor<platform::GPUPlace, float>;
template class Col2VolFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
