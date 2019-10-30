// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Part of the following code in this file refs to
// https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/nn/modulated_deformable_im2col.cuh
//
// Copyright (c) 2018 Microsoft
// Licensed under The MIT License [see LICENSE for details]
// \file modulated_deformable_im2col.cuh
// \brief
// \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/deformable_conv_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__device__ T DmcnGetGradientWeight(T argmax_h, T argmax_w, const int h,
                                   const int w, const int height,
                                   const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
__global__ void ModulatedDeformableCol2imGpuKernel(
    const int nthreads, const T* data_col, const T* data_offset,
    const T* data_mask, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int deformable_group, const int height_col,
    const int width_col, T* grad_im) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t thread = index; thread < nthreads; thread += offset) {
    const int j = (thread / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (thread / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        thread / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = thread % width_col;
    int h_out = (thread / width_col) % height_col;
    int b = (thread / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T* data_offset_ptr = data_offset +
                               (b * deformable_group + deformable_group_index) *
                                   2 * kernel_h * kernel_w * height_col *
                                   width_col;
    const T* data_mask_ptr = data_mask +
                             (b * deformable_group + deformable_group_index) *
                                 kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[thread] * mask;
    const int cur_h = static_cast<int>(cur_inv_h_data);
    const int cur_w = static_cast<int>(cur_inv_w_data);
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight =
              DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                    cur_w + dx, height, width);

          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename T>
inline void ModulatedDeformableCol2im(
    const platform::DeviceContext& ctx, const T* data_col, const T* data_offset,
    const T* data_mask, const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> kernel_shape, const std::vector<int> pad,
    const std::vector<int> stride, const std::vector<int> dilation,
    const int deformable_group, T* grad_im) {
  int channel_per_deformable_group = im_shape[0] / deformable_group;
  int num_kernels = col_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];
  int blocks = NumBlocks(num_kernels);
  int threads = kNumCUDAThreads;

  ModulatedDeformableCol2imGpuKernel<T><<<
      blocks, threads, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      num_kernels, data_col, data_offset, data_mask, im_shape[0], im_shape[1],
      im_shape[2], kernel_shape[2], kernel_shape[3], pad[0], pad[1], stride[0],
      stride[1], dilation[0], dilation[1], channel_per_deformable_group,
      col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im);
}

template <typename T>
__device__ T DmcnGetCoordinateWeight(T argmax_h, T argmax_w, const int height,
                                     const int width, const T* im_data,
                                     const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }
  return weight;
}

template <typename T>
__device__ T DmcnIm2colBilinear(const T* bottom_data, const int data_width,
                                const int height, const int width, T h, T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void ModulatedDeformableCol2imCoordGpuKernel(
    const int nthreads, const T* data_col, const T* data_im,
    const T* data_offset, const T* data_mask, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int offset_channels, const int deformable_group, const int height_col,
    const int width_col, T* grad_offset, T* grad_mask) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    T val = 0, mval = 0;
    const int w = i % width_col;
    const int h = (i / width_col) % height_col;
    const int c = (i / width_col / height_col) % offset_channels;
    const int b = (i / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T* data_col_ptr = data_col +
                            deformable_group_index *
                                channel_per_deformable_group * batch_size *
                                width_col * height_col;
    const T* data_im_ptr = data_im +
                           (b * deformable_group + deformable_group_index) *
                               channel_per_deformable_group / kernel_h /
                               kernel_w * height * width;
    const T* data_offset_ptr = data_offset +
                               (b * deformable_group + deformable_group_index) *
                                   2 * kernel_h * kernel_w * height_col *
                                   width_col;
    const T* data_mask_ptr = data_mask +
                             (b * deformable_group + deformable_group_index) *
                                 kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = offset_c / 2; col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      const T mask = data_mask_ptr[data_mask_hw_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] *
                DmcnIm2colBilinear(data_im_ptr + cnt * height * width, width,
                                   height, width, inv_h, inv_w);
      }
      const T weight = DmcnGetCoordinateWeight(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    grad_offset[i] = val;
    if (offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
  }
}

template <typename T>
inline void ModulatedDeformableCol2imCoord(
    const platform::DeviceContext& ctx, const T* data_col, const T* data_im,
    const T* data_offset, const T* data_mask,
    const std::vector<int64_t> im_shape, const std::vector<int64_t> col_shape,
    const std::vector<int64_t> kernel_shape, const std::vector<int> paddings,
    const std::vector<int> strides, const std::vector<int> dilations,
    const int deformable_groups, T* grad_offset, T* grad_mask) {
  int num_kernels = 2 * kernel_shape[2] * kernel_shape[3] * col_shape[1] *
                    col_shape[2] * col_shape[3] * deformable_groups;
  int channel_per_deformable_group = col_shape[0] / deformable_groups;
  int blocks = NumBlocks(num_kernels);
  int threads = kNumCUDAThreads;

  ModulatedDeformableCol2imCoordGpuKernel<T><<<
      blocks, threads, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      num_kernels, data_col, data_im, data_offset, data_mask, im_shape[0],
      im_shape[1], im_shape[2], kernel_shape[2], kernel_shape[3], paddings[0],
      paddings[1], strides[0], strides[1], dilations[0], dilations[1],
      channel_per_deformable_group, col_shape[1],
      2 * kernel_shape[2] * kernel_shape[3] * deformable_groups,
      deformable_groups, col_shape[2], col_shape[3], grad_offset, grad_mask);
}

template <typename T>
__global__ void ModulatedDeformableIm2colGpuKernel(
    const int nthreads, const T* data_im, const T* data_offset,
    const T* data_mask, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T* data_col) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    const int w_col = i % width_col;
    const int h_col = (i / width_col) % height_col;
    const int b_col = (i / width_col) / height_col % batch_size;
    const int c_im = (i / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const T* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
inline void ModulatedDeformableIm2col(
    const platform::DeviceContext& ctx, const T* data_im, const T* data_offset,
    const T* data_mask, const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> filter_shape, const std::vector<int> paddings,
    const std::vector<int> strides, const std::vector<int> dilations,
    const int deformable_groups, T* data_col) {
  int channel_per_deformable_group = im_shape[0] / deformable_groups;
  int num_kernels = im_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];

  int blocks = NumBlocks(num_kernels);
  int threads = kNumCUDAThreads;

  ModulatedDeformableIm2colGpuKernel<T><<<
      blocks, threads, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      num_kernels, data_im, data_offset, data_mask, im_shape[1], im_shape[2],
      filter_shape[2], filter_shape[3], paddings[0], paddings[1], strides[0],
      strides[1], dilations[0], dilations[1], channel_per_deformable_group,
      col_shape[1], im_shape[0], deformable_groups, col_shape[2], col_shape[3],
      data_col);
}

template <typename T>
__global__ void FilterGradAddupGpuKernel(const int nthreads, const int n,
                                         const int height, const int width,
                                         const T* dweight_3d, T* filter_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    filter_grad[i] = filter_grad[i] + dweight_3d[i];
  }
}

template <typename DeviceContext, typename T>
class DeformableConvCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor offset = *ctx.Input<Tensor>("Offset");
    const Tensor mask = *ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.cuda_device_context();

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(input->dims()[0]);

    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    // col_shape_vec: {c_i * k_h * k_w, im2col_step, o_h, o_w}
    std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
    col_buffer_shape_vec[0] =
        input->dims()[1] * filter.dims()[2] * filter.dims()[3];
    col_buffer_shape_vec[1] = im2col_step;
    for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
      col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_buffer_shape_vec));
    std::vector<int64_t> output_buffer_shape_vec(1);
    output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                                 output_shape_vec[2] * output_shape_vec[3];
    framework::DDim output_shape(framework::make_ddim(output_buffer_shape_vec));
    Tensor col_buffer;
    Tensor output_buffer;
    col_buffer = ctx.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
    output_buffer =
        ctx.AllocateTmpTensor<T, DeviceContext>(output_shape, dev_ctx);

    int64_t M = output_shape_vec[1] / groups;
    int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
    int64_t K =
        input->dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;

    Tensor weight_3d;
    weight_3d.ShareDataWith(filter).Resize(
        framework::make_ddim({groups, M, K}));
    Tensor col_buffer_3d;
    col_buffer_3d.ShareDataWith(col_buffer)
        .Resize(framework::make_ddim({groups, K, N}));
    Tensor output_4d;
    output_4d.ShareDataWith(output_buffer)
        .Resize(framework::make_ddim({batch_size / im2col_step, groups, M, N}));
    output_4d.mutable_data<T>(ctx.GetPlace());
    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());
    std::vector<int64_t> input_shape_vec = framework::vectorize(input_shape);

    int input_dim = input->numel() / input->dims()[0];
    int input_offset_dim = offset.numel() / offset.dims()[0];
    int input_mask_dim = mask.numel() / mask.dims()[0];

    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    const T* input_ptr = input->data<T>();
    const T* offset_ptr = offset.data<T>();
    const T* mask_ptr = mask.data<T>();
    col_buffer.mutable_data<T>(ctx.GetPlace());
    T* col_buffer_ptr = col_buffer.data<T>();

    for (int i = 0; i < batch_size / im2col_step; ++i) {
      ModulatedDeformableIm2col(
          ctx.device_context(), input_ptr + i * im2col_step * input_dim,
          offset_ptr + i * im2col_step * input_offset_dim,
          mask_ptr + i * im2col_step * input_mask_dim, input_shape_vec,
          col_buffer_shape_vec, filter_shape_vec, paddings, strides, dilations,
          deformable_groups, col_buffer_ptr);

      Tensor output_3d = output_4d.Slice(i, i + 1).Resize(
          framework::slice_ddim(output_4d.dims(), 1, output_4d.dims().size()));
      for (int g = 0; g < groups; ++g) {
        Tensor weight_3d_slice =
            weight_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                weight_3d.dims(), 1, weight_3d.dims().size()));
        Tensor col_buffer_3d_slice =
            col_buffer_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
        Tensor output_3d_slice =
            output_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                output_3d.dims(), 1, output_3d.dims().size()));

        blas.MatMul(weight_3d_slice, false, col_buffer_3d_slice, false, T(1.0),
                    &output_3d_slice, T(0.0));
      }
    }
    output->ShareDataWith(output_buffer)
        .Resize(framework::make_ddim(output_shape_vec));
  }
};

template <typename DeviceContext, typename T>
class DeformableConvGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor* offset_grad = ctx.Output<Tensor>(framework::GradVarName("Offset"));
    Tensor* mask_grad = ctx.Output<Tensor>(framework::GradVarName("Mask"));

    const Tensor* input = ctx.Input<Tensor>("Input");
    Tensor offset = *ctx.Input<Tensor>("Offset");
    Tensor mask = *ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");
    if (!input_grad && !filter_grad && !offset_grad && !mask_grad) return;

    int groups = ctx.Attr<int>("groups");
    int deformable_groups = ctx.Attr<int>("deformable_groups");
    int im2col_step = ctx.Attr<int>("im2col_step");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    auto& dev_ctx = ctx.cuda_device_context();
    const int batch_size = static_cast<int>(input->dims()[0]);

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());
    std::vector<int64_t> input_shape_vec = framework::vectorize(input_shape);
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(output_grad->dims()));

    std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
    col_buffer_shape_vec[0] =
        input->dims()[1] * filter.dims()[2] * filter.dims()[3];
    col_buffer_shape_vec[1] = im2col_step;
    for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
      col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_buffer_shape_vec));
    std::vector<int64_t> output_buffer_shape_vec(1);
    output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                                 output_shape_vec[2] * output_shape_vec[3];
    framework::DDim output_shape(framework::make_ddim(output_buffer_shape_vec));
    Tensor col_buffer;
    Tensor output_buffer;
    col_buffer = ctx.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
    output_buffer =
        ctx.AllocateTmpTensor<T, DeviceContext>(output_shape, dev_ctx);

    output_buffer.ShareDataWith(*output_grad);

    int64_t M =
        input_shape_vec[0] / groups * filter_shape_vec[2] * filter_shape_vec[3];
    int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
    int64_t K = output_shape_vec[1] / groups;

    framework::DDim weight_3d_shape = {groups, K, M};
    framework::DDim out_grad_4d_shape = {batch_size / im2col_step, groups, K,
                                         N};
    framework::DDim col_buffer_3d_shape = {groups, M, N};
    framework::DDim filter_grad_shape = {groups, K, M};

    Tensor weight_3d;
    weight_3d.ShareDataWith(filter).Resize(weight_3d_shape);
    Tensor out_grad_4d;
    out_grad_4d.ShareDataWith(output_buffer).Resize(out_grad_4d_shape);
    Tensor col_buffer_3d;
    col_buffer_3d.ShareDataWith(col_buffer).Resize(col_buffer_3d_shape);

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    col_buffer.mutable_data<T>(ctx.GetPlace());
    col_buffer_3d.mutable_data<T>(ctx.GetPlace());
    out_grad_4d.mutable_data<T>(ctx.GetPlace());

    int input_dim = input->numel() / input->dims()[0];
    int input_offset_dim = offset.numel() / offset.dims()[0];
    int input_mask_dim = mask.numel() / mask.dims()[0];

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      filter_grad->Resize(filter_grad_shape);
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
    }

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, input_grad, static_cast<T>(0));
    }

    if (offset_grad && mask_grad) {
      offset_grad->mutable_data<T>(ctx.GetPlace());
      mask_grad->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, offset_grad, static_cast<T>(0));
      set_zero(dev_ctx, mask_grad, static_cast<T>(0));
    }

    for (int i = 0; i < batch_size / im2col_step; ++i) {
      Tensor out_grad_3d =
          out_grad_4d.Slice(i, i + 1).Resize(framework::slice_ddim(
              out_grad_4d.dims(), 1, out_grad_4d.dims().size()));
      for (int g = 0; g < groups; ++g) {
        Tensor weight_3d_slice =
            weight_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                weight_3d.dims(), 1, weight_3d.dims().size()));
        Tensor out_grad_3d_slice =
            out_grad_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                out_grad_3d.dims(), 1, out_grad_3d.dims().size()));
        Tensor col_buffer_3d_slice =
            col_buffer_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));

        blas.MatMul(weight_3d_slice, true, out_grad_3d_slice, false, T(1.0),
                    &col_buffer_3d_slice, T(0.0));
      }
      col_buffer.Resize(col_shape);

      T* col_buffer_ptr = col_buffer.data<T>();
      const T* input_ptr = input->data<T>();
      const T* offset_ptr = offset.data<T>();
      const T* mask_ptr = mask.data<T>();

      if (mask_grad && offset_grad) {
        T* offset_grad_ptr = offset_grad->data<T>();
        T* mask_grad_ptr = mask_grad->data<T>();
        ModulatedDeformableCol2imCoord(
            ctx.device_context(), col_buffer_ptr,
            input_ptr + i * im2col_step * input_dim,
            offset_ptr + i * im2col_step * input_offset_dim,
            mask_ptr + i * im2col_step * input_mask_dim, input_shape_vec,
            col_buffer_shape_vec, filter_shape_vec, paddings, strides,
            dilations, deformable_groups,
            offset_grad_ptr + i * im2col_step * input_offset_dim,
            mask_grad_ptr + i * im2col_step * input_mask_dim);
      }
      if (input_grad) {
        T* input_grad_ptr = input_grad->data<T>();
        ModulatedDeformableCol2im(
            ctx.device_context(), col_buffer_ptr,
            offset_ptr + i * im2col_step * input_offset_dim,
            mask_ptr + i * im2col_step * input_mask_dim, input_shape_vec,
            col_buffer_shape_vec, filter_shape_vec, paddings, strides,
            dilations, deformable_groups,
            input_grad_ptr + i * im2col_step * input_dim);
        input_grad->Resize(input->dims());
      }

      ModulatedDeformableIm2col(
          ctx.device_context(), input_ptr + i * im2col_step * input_dim,
          offset_ptr + i * im2col_step * input_offset_dim,
          mask_ptr + i * im2col_step * input_mask_dim, input_shape_vec,
          col_buffer_shape_vec, filter_shape_vec, paddings, strides, dilations,
          deformable_groups, col_buffer_ptr);

      col_buffer_3d.Resize(col_buffer_3d_shape);

      if (filter_grad) {
        Tensor dweight_3d;
        dweight_3d =
            ctx.AllocateTmpTensor<T, DeviceContext>(filter_grad_shape, dev_ctx);
        for (int g = 0; g < groups; ++g) {
          Tensor out_grad_3d_slice =
              out_grad_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                  out_grad_3d.dims(), 1, out_grad_3d.dims().size()));
          Tensor col_buffer_3d_slice =
              col_buffer_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                  col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
          Tensor dweight_3d_slice =
              dweight_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                  dweight_3d.dims(), 1, dweight_3d.dims().size()));

          blas.MatMul(out_grad_3d_slice, false, col_buffer_3d_slice, true,
                      T(1.0), &dweight_3d_slice, T(0.0));
        }
        FilterGradAddupGpuKernel<
            T><<<NumBlocks(dweight_3d.numel()), kNumCUDAThreads, 0,
                 ctx.cuda_device_context().stream()>>>(
            dweight_3d.numel(), groups, K, M, dweight_3d.data<T>(),
            filter_grad->data<T>());
      }
    }
    if (filter_grad) {
      filter_grad->Resize(filter.dims());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(deformable_conv,
                        ops::DeformableConvCUDAKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(deformable_conv_grad,
                        ops::DeformableConvGradCUDAKernel<CUDA, float>);
