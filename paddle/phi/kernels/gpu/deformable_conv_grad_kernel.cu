// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.   
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/deformable_conv_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/deformable_conv_grad_kernel_impl.h"

namespace phi {
#define THRESHOLD 26624
static constexpr int kNumCUDAThreadsS = 64;
static constexpr int kNumCUDAThreadsL = 256;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  int NumThreads = N >= THRESHOLD ? kNumCUDAThreadsL : kNumCUDAThreadsS;
  return std::min((N + NumThreads - 1) / NumThreads,
                  kNumMaximumNumBlocks);
}

#define INT_BITS 32
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T& operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T& operator[](int i) { return val[i]; }
};

struct FastDivMod {
  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = phi::AlignedVector<uint32_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor) break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  int32_t shift_val;
  uint32_t divisor;
  uint32_t multiplier;
};

template <typename T>
__global__ void ModulatedDeformableCol2imGpuKernel(
    const int nthreads,
    const T* data_col,
    const T* data_offset,
    const T* data_mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int deformable_group,
    const int height_col,
    const int width_col,
    T* grad_im,
    const FastDivMod width_col_r,
    const FastDivMod height_col_r,
    const FastDivMod batch_size_r,
    const FastDivMod kernel_w_r,
    const FastDivMod kernel_h_r,
    const FastDivMod channel_per_deformable_group_r) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t thread = index; thread < nthreads; thread += offset) {
    const int tmp0 = width_col_r.Div(thread);
    const int tmp1 = height_col_r.Div(tmp0);
    const int tmp2 = batch_size_r.Div(tmp1);
    const int tmp3 = kernel_w_r.Div(tmp2);
    const int tmp4 = kernel_h_r.Div(tmp3);
    const int j = tmp2 - kernel_w * tmp3;
    const int i = tmp3 - kernel_h * tmp4;
    const int c = tmp4;
    const int deformable_group_index = channel_per_deformable_group_r.Div(c);
    int w_out = thread - width_col * tmp0;
    int h_out = tmp0 - height_col * tmp1;
    int b = tmp1 - batch_size * tmp2;
    const int data_mask_hw_ptr =
    ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    T mask = static_cast<T>(1);
    if (data_mask) {
      const T* data_mask_ptr =
          data_mask + (b * deformable_group + deformable_group_index) *
                          kernel_h * kernel_w * height_col * width_col;
      mask = data_mask_ptr[data_mask_hw_ptr];
    }
    T cur_top_grad = data_col[thread] * mask;
    const T* data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const int cur_h = static_cast<int>(cur_inv_h_data);
    const int cur_w = static_cast<int>(cur_inv_w_data);
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight = DmcnGetGradientWeight(cur_inv_h_data,
                                           cur_inv_w_data,
                                           cur_h + dy,
                                           cur_w + dx,
                                           height,
                                           width);

          phi::CudaAtomicAdd(grad_im + cur_bottom_grad_pos,
                             weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename T, typename Context>
void ModulatedDeformableCol2im(const Context& dev_ctx,
                               const T* data_col,
                               const T* data_offset,
                               const T* data_mask,
                               const std::vector<int64_t>& im_shape,
                               const std::vector<int64_t>& col_shape,
                               const std::vector<int64_t>& kernel_shape,
                               const std::vector<int>& pad,
                               const std::vector<int>& stride,
                               const std::vector<int>& dilation,
                               const int deformable_group,
                               T* grad_im) {
  int channel_per_deformable_group = im_shape[0] / deformable_group;
  int num_kernels = col_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];
  int blocks = NumBlocks(num_kernels);
  int threads = num_kernels >= THRESHOLD ? kNumCUDAThreadsL : kNumCUDAThreadsS;
  FastDivMod width_col_r = FastDivMod(col_shape[3]);
  FastDivMod height_col_r = FastDivMod(col_shape[2]);
  FastDivMod batch_size_r = FastDivMod(col_shape[1]);
  FastDivMod kernel_w_r = FastDivMod(kernel_shape[3]);
  FastDivMod kernel_h_r = FastDivMod(kernel_shape[2]);
  FastDivMod channel_per_deformable_group_r = FastDivMod(channel_per_deformable_group);
  ModulatedDeformableCol2imGpuKernel<T>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(num_kernels,
                                                 data_col,
                                                 data_offset,
                                                 data_mask,
                                                 im_shape[0],
                                                 im_shape[1],
                                                 im_shape[2],
                                                 kernel_shape[2],
                                                 kernel_shape[3],
                                                 pad[0],
                                                 pad[1],
                                                 stride[0],
                                                 stride[1],
                                                 dilation[0],
                                                 dilation[1],
                                                 channel_per_deformable_group,
                                                 col_shape[1],
                                                 deformable_group,
                                                 col_shape[2],
                                                 col_shape[3],
                                                 grad_im,
                                                 width_col_r,
                                                 height_col_r,
                                                 batch_size_r,
                                                 kernel_w_r,
                                                 kernel_h_r,
                                                 channel_per_deformable_group_r);
}

template <typename T>
__global__ void ModulatedDeformableCol2imCoordGpuKernel(
    const int nthreads,
    const T* data_col,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int offset_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    T* grad_offset,
    T* grad_mask,
    FastDivMod width_col_r,
    FastDivMod height_col_r,
    FastDivMod batch_size_r,
    FastDivMod kernel_w_r,
    FastDivMod kernel_h_r,
    FastDivMod offset_channels_r,
    FastDivMod kernel_h_kernel_w_r) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t i = index; i < nthreads; i += offset) {
    T val = 0, mval = 0;
    const int temp0 = width_col_r.Div(i);
    const int temp1 = height_col_r.Div(temp0);
    const int w = i - temp0 * width_col;
    const int h = temp0 - temp1 * height_col;
    const int b = offset_channels_r.Div(temp1);
    const int c = temp1 - b * offset_channels;
    const int deformable_group_index = kernel_h_kernel_w_r.Div(c) >> 1;
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T* data_col_ptr = data_col + deformable_group_index *
                                           channel_per_deformable_group *
                                           batch_size * width_col * height_col;
    const T* data_im_ptr =
       data_im + kernel_h_kernel_w_r.Div((b * deformable_group + deformable_group_index) *
                      channel_per_deformable_group) *
                      height * width;
    const T* data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T* data_mask_ptr =
        data_mask
            ? data_mask + (b * deformable_group + deformable_group_index) *
                              kernel_h * kernel_w * height_col * width_col
            : nullptr;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;
    T offset_h;
    int data_offset_h_ptr_old = -1;
    T offset_w;
    int data_offset_w_ptr_old = -1;
    T mask;
    int data_mask_hw_ptr_old = -1;

    for (int col_c = offset_c / 2; col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      const int tmp0 = width_col_r.Div(col_pos);
      const int tmp1 = height_col_r.Div(tmp0);
      const int tmp2 = batch_size_r.Div(tmp1);
      const int tmp3 = kernel_w_r.Div(tmp2);
      int j = tmp2 - tmp3 * kernel_w;
      int i = tmp3 - kernel_h_r.Div(tmp3) * kernel_h;
      int w_out = col_pos - tmp0 * width_col;
      int h_out = tmp0 - tmp1 * height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      if(data_offset_h_ptr_old != data_offset_h_ptr)
      {
        offset_h = data_offset_ptr[data_offset_h_ptr];
        data_offset_h_ptr_old = data_offset_h_ptr;
      }
      if(data_offset_w_ptr_old != data_offset_w_ptr)
      {
        offset_w = data_offset_ptr[data_offset_w_ptr];
        data_offset_w_ptr_old = data_offset_w_ptr;
      }
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      T data_col_value = data_col_ptr[col_pos];
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_value *
                funcs::DmcnIm2colBilinear(data_im_ptr + cnt * height * width,
                                          width,
                                          height,
                                          width,
                                          inv_h,
                                          inv_w);
      }
      const T weight =
          DmcnGetCoordinateWeight(inv_h,
                                  inv_w,
                                  height,
                                  width,
                                  data_im_ptr + cnt * height * width,
                                  width,
                                  bp_dir);
      if (data_mask_ptr) {
        const int data_mask_hw_ptr =
            (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
        if(data_mask_hw_ptr_old != data_mask_hw_ptr)
        {
          mask = data_mask_ptr[data_mask_hw_ptr];
          data_mask_hw_ptr_old = data_mask_hw_ptr;
        }
        val += weight * data_col_value * mask;
      } else {
        val += weight * data_col_value;
      }
      cnt += 1;
    }
    grad_offset[i] = val;
    if (grad_mask && offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
  }
}

template <typename T, typename Context>
void ModulatedDeformableCol2imCoord(const Context& dev_ctx,
                                    const T* data_col,
                                    const T* data_im,
                                    const T* data_offset,
                                    const T* data_mask,
                                    const std::vector<int64_t>& im_shape,
                                    const std::vector<int64_t>& col_shape,
                                    const std::vector<int64_t>& kernel_shape,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& dilations,
                                    const int deformable_groups,
                                    T* grad_offset,
                                    T* grad_mask) {
  int num_kernels = 2 * kernel_shape[2] * kernel_shape[3] * col_shape[1] *
                    col_shape[2] * col_shape[3] * deformable_groups;
  int channel_per_deformable_group = col_shape[0] / deformable_groups;
  int blocks = NumBlocks(num_kernels);
  int threads = num_kernels >= THRESHOLD ? kNumCUDAThreadsL : kNumCUDAThreadsS;
  FastDivMod width_col_r = FastDivMod(col_shape[3]);
  FastDivMod height_col_r = FastDivMod(col_shape[2]);
  FastDivMod batch_size_r = FastDivMod(col_shape[1]);
  FastDivMod kernel_w_r = FastDivMod(kernel_shape[3]);
  FastDivMod kernel_h_r = FastDivMod(kernel_shape[2]);
  FastDivMod offset_channels_r = FastDivMod(2 * kernel_shape[2] * kernel_shape[3] * deformable_groups);
  FastDivMod kernel_h_kernel_w_r = FastDivMod(kernel_shape[2]*kernel_shape[3]);
  ModulatedDeformableCol2imCoordGpuKernel<T>
      <<<blocks, threads, 0, dev_ctx.stream()>>>(
          num_kernels,
          data_col,
          data_im,
          data_offset,
          data_mask,
          im_shape[0],
          im_shape[1],
          im_shape[2],
          kernel_shape[2],
          kernel_shape[3],
          paddings[0],
          paddings[1],
          strides[0],
          strides[1],
          dilations[0],
          dilations[1],
          channel_per_deformable_group,
          col_shape[1],
          2 * kernel_shape[2] * kernel_shape[3] * deformable_groups,
          deformable_groups,
          col_shape[2],
          col_shape[3],
          grad_offset,
          grad_mask,
          width_col_r,
          height_col_r,
          batch_size_r,
          kernel_w_r,
          kernel_h_r,
          offset_channels_r,
          kernel_h_kernel_w_r);
}

template <typename T>
__global__ void FilterGradAddupGpuKernel(const int nthreads,
                                         const int n,
                                         const int height,
                                         const int width,
                                         const T* dweight_3d,
                                         T* filter_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;

  constexpr int VecSize = sizeof(int4) / sizeof(T);
  if (sizeof(int4) > sizeof(T)) {
    int4 in1, in2;
    T* in1_ptr = reinterpret_cast<T*>(&in1);
    T* in2_ptr = reinterpret_cast<T*>(&in2);
    constexpr int VecSize = sizeof(int4) / sizeof(T);
#pragma unroll
    for (int i = index; i < nthreads / VecSize; i += offset) {
      in1 = reinterpret_cast<const int4*>(dweight_3d)[i];
      in2 = reinterpret_cast<int4*>(filter_grad)[i];
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        in2_ptr[j] += in1_ptr[j];
      }
      reinterpret_cast<int4*>(filter_grad)[i] = in2;
    }

    // in only one thread, process final elements (if there are any)
    if (index == nthreads / VecSize) {
#pragma unroll
      for (int i = index * VecSize; i < nthreads; ++i)
        filter_grad[i] = filter_grad[i] + dweight_3d[i];
    }
  } else {
    for (size_t i = index; i < nthreads; i += offset) {
      filter_grad[i] = filter_grad[i] + dweight_3d[i];
    }
  }
}

template <typename T, typename Context>
void FilterGradAddup(const Context& dev_ctx,
                     const int nthreads,
                     const int n,
                     const int height,
                     const int width,
                     const T* dweight_3d,
                     T* filter_grad) {
  const int kVecSize = sizeof(int4) / sizeof(T);
  const int block = 512;
  const int num_per_block = kVecSize * block;
  const int grid = (nthreads + num_per_block - 1) / num_per_block;
  FilterGradAddupGpuKernel<T>
      <<<grid, block, 0, dev_ctx.stream()>>>(
          nthreads, n, height, width, dweight_3d, filter_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(deformable_conv_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DeformableConvGradKernel,
                   float,
                   double) {}
