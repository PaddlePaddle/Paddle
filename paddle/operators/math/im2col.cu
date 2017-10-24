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
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void im2col(const T* data_im, int num_outs, int height, int width,
                       int filter_height, int filter_width, int stride_height,
                       int stride_width, int padding_height, int padding_width,
                       int output_height, int output_width, T* data_col) {
  int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < num_outs) {
    int w_out = index % output_width;
    index /= output_width;
    int h_out = index % output_height;
    int channel_in = index / output_height;
    int channel_out = channel_in * filter_height * filter_width;
    int h_in = h_out * stride_height;
    int w_in = w_out * stride_width;

    data_col += (channel_out * output_height + h_out) * output_width + w_out;
    for (int i = 0; i < filter_height; ++i) {
      for (int j = 0; j < filter_width; ++j) {
        int rIdx = int(h_in + i);
        int cIdx = int(w_in + j);
        if ((rIdx - (int)padding_height) >= (int)height ||
            (rIdx - (int)padding_height) < 0 ||
            (cIdx - (int)padding_width) >= (int)width ||
            (cIdx - (int)padding_width) < 0) {
          *data_col = 0;
        } else {
          rIdx = rIdx + channel_in * height - padding_height;
          cIdx = cIdx - padding_width;
          *data_col = data_im[rIdx * width + cIdx];
        }
        data_col += output_height * output_width;
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                    platform::GPUPlace, T> {
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

    PADDLE_ENFORCE((input_height + padding_up + padding_down - filter_height) /
                           stride_height +
                       1 ==
                   output_height);
    PADDLE_ENFORCE((input_width + padding_left + padding_right - filter_width) /
                           stride_width +
                       1 ==
                   output_width);

    int num_outputs = input_channels * output_height * output_width;
    int blocks = (num_outputs + 1024 - 1) / 1024;
    int block_x = 512;
    int block_y = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(block_x, block_y);
    im2col<T><<<grid, threads, 0,
                reinterpret_cast<const platform::CUDADeviceContext&>(context)
                    .stream()>>>(
        im.data<T>(), num_outputs, input_height, input_width, filter_height,
        filter_width, stride_height, stride_width, padding_up, padding_left,
        output_height, output_width, col.data<T>());
  }
};

template <class T>
__global__ void col2im(size_t n, const T* data_col, size_t height, size_t width,
                       size_t channels, size_t filter_height,
                       size_t filter_width, size_t stride_height,
                       size_t stride_width, size_t padding_height,
                       size_t padding_width, size_t output_height,
                       size_t output_width, T* data_im) {
  size_t index =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < n) {
    T val = 0;
    int w = int(index % width);
    int h = int((index / width) % height);
    int c = int(index / (width * height));
    if ((w - (int)padding_width) >= 0 &&
        (w - (int)padding_width) < (width - 2 * padding_width) &&
        (h - (int)padding_height) >= 0 &&
        (h - padding_height) < (height - 2 * padding_height)) {
      // compute the start and end of the output
      int w_col_start = (w < (int)filter_width)
                            ? 0
                            : (w - int(filter_width)) / (int)stride_width + 1;
      int w_col_end =
          min((int)(w / (int)stride_width + 1), (int)(output_width));
      int h_col_start = (h < (int)filter_height)
                            ? 0
                            : (h - (int)filter_height) / (int)stride_height + 1;
      int h_col_end = min(int(h / stride_height + 1), int(output_height));
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          // the col location: [c * width * height + h_out, w_out]
          int c_col = int(c * filter_height * filter_width) +
                      (h - h_col * (int)stride_height) * (int)filter_width +
                      (w - w_col * (int)stride_width);
          val +=
              data_col[(c_col * output_height + h_col) * output_width + w_col];
        }
      }
      h -= padding_height;
      w -= padding_width;
      data_im[c * ((width - 2 * padding_width) *
                   (height - 2 * padding_height)) +
              h * (width - 2 * padding_width) + w] += val;
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                    platform::GPUPlace, T> {
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

    PADDLE_ENFORCE((input_height + padding_up + padding_down - filter_height) /
                           stride_height +
                       1 ==
                   output_height);
    PADDLE_ENFORCE((input_width + padding_left + padding_right - filter_width) /
                           stride_width +
                       1 ==
                   output_width);

    size_t num_kernels = input_channels *
                         (input_height + padding_up + padding_down) *
                         (input_width + padding_left + padding_right);

    size_t blocks = (num_kernels + 1024 - 1) / 1024;
    size_t block_x = 512;
    size_t block_y = (blocks + 512 - 1) / 512;
    dim3 threads(1024, 1);
    dim3 grid(block_x, block_y);

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    col2im<T><<<grid, threads, 0,
                reinterpret_cast<const platform::CUDADeviceContext&>(context)
                    .stream()>>>(
        num_kernels, col.data<T>(), input_height + padding_up + padding_down,
        input_width + padding_left + padding_left, input_channels,
        filter_height, filter_width, stride_height, stride_width, padding_up,
        padding_left, output_height, output_width, im.data<T>());
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::GPUPlace, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::GPUPlace, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::GPUPlace, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::GPUPlace, double>;

template <class T>
__global__ void im2colOCF(const T* im_data, T* col_data, int input_channels,
                          int input_height, int input_width, int filter_height,
                          int filter_width, int stride_height, int stride_width,
                          int padding_height, int padding_width,
                          int output_height, int output_width) {
  int swid = blockIdx.x;
  int shid = blockIdx.y;
  for (int channelid = threadIdx.z; channelid < input_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * input_width +
                        channelid * input_height * input_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * output_width + swid) *
                             (input_channels * filter_height * filter_width);

        if (height_offset >= input_height || height_offset < 0 ||
            width_offset >= input_width || width_offset < 0) {
          col_data[col_offset] = T(0);
        } else {
          col_data[col_offset] = im_data[im_offset];
        }
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                    platform::GPUPlace, T> {
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

    PADDLE_ENFORCE((input_height + padding_up + padding_down - filter_height) /
                           stride_height +
                       1 ==
                   output_height);
    PADDLE_ENFORCE((input_width + padding_left + padding_right - filter_width) /
                           stride_width +
                       1 ==
                   output_width);

    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y,
                 std::min(block_dim_z, input_channels));
    dim3 grid(output_width, output_height);
    im2colOCF<T><<<grid, threads, 0,
                   reinterpret_cast<const platform::CUDADeviceContext&>(context)
                       .stream()>>>(
        im.data<T>(), col.data<T>(), input_channels, input_height, input_width,
        filter_height, filter_width, stride_height, stride_width, padding_up,
        padding_left, output_height, output_width);
  }
};

template <class T>
__global__ void col2imOCF(T* im_data, const T* col_data, int input_channels,
                          int input_height, int input_width, int filter_height,
                          int filter_width, int stride_height, int stride_width,
                          int padding_height, int padding_width,
                          int output_height, int output_width) {
  int swid = blockIdx.x;
  int shid = blockIdx.y;
  for (int channelid = threadIdx.z; channelid < input_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * input_width +
                        channelid * input_height * input_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * output_width + swid) *
                             (input_channels * filter_height * filter_width);

        if (height_offset >= 0 && height_offset < input_height &&
            width_offset >= 0 && width_offset < input_width) {
          paddle::platform::CudaAtomicAdd(im_data + im_offset,
                                          col_data[col_offset]);
        }
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                    platform::GPUPlace, T> {
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

    PADDLE_ENFORCE((input_height + padding_up + padding_down - filter_height) /
                           stride_height +
                       1 ==
                   output_height);
    PADDLE_ENFORCE((input_width + padding_left + padding_right - filter_width) /
                           stride_width +
                       1 ==
                   output_width);

    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y,
                 std::min(block_dim_z, input_channels));
    dim3 grid(output_width, output_height);
    col2imOCF<T><<<grid, threads, 0,
                   reinterpret_cast<const platform::CUDADeviceContext&>(context)
                       .stream()>>>(
        im.data<T>(), col.data<T>(), input_channels, input_height, input_width,
        filter_height, filter_width, stride_height, stride_width, padding_up,
        padding_left, output_height, output_width);
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::GPUPlace, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::GPUPlace, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::GPUPlace, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
