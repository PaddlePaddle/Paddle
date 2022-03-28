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

#pragma once

#include "glog/logging.h"
#include "paddle/fluid/memory/buffer.h"
#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

template <typename T1, typename T2, typename Functor, int VecSize>
static __global__ void VectorizedUnaryCUDAKernel(const T1 *x, T2 *y, int n,
                                                 Functor functor) {
  int i = (threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = blockDim.x * gridDim.x * VecSize;
  int max_vec_n = n - VecSize;

  for (; i <= max_vec_n; i += stride) {
    phi::AlignedVector<T1, VecSize> x_vec;
    phi::AlignedVector<T2, VecSize> y_vec;
    phi::Load(x + i, &x_vec);
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      y_vec[j] = functor(x_vec[j]);
    }
    phi::Store(y_vec, y + i);
  }

  for (; i < n; ++i) {
    auto tmp_x = x[i];
    y[i] = functor(x[i]);
  }
}

template <typename T1, typename T2, typename T3, typename Functor, int VecSize>
static __global__ void VectorizedBinaryCUDAKernel(const T1 *x, const T2 *y,
                                                  T3 *z, int n,
                                                  Functor functor) {
  int i = (threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = blockDim.x * gridDim.x * VecSize;
  int max_vec_n = n - VecSize;

  for (; i <= max_vec_n; i += stride) {
    phi::AlignedVector<T1, VecSize> x_vec;
    phi::AlignedVector<T2, VecSize> y_vec;
    phi::AlignedVector<T3, VecSize> z_vec;
    phi::Load(x + i, &x_vec);
    phi::Load(y + i, &y_vec);

#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      z_vec[j] = functor(x_vec[j], y_vec[j]);
    }
    phi::Store(z_vec, z + i);
  }

  for (; i < n; ++i) {
    z[i] = functor(x[i], y[i]);
  }
}

template <typename T1, typename T2, typename Functor>
static void LaunchVectorizedUnaryFunctorKernel(
    const platform::CUDADeviceContext &dev_ctx, const T1 *x, T2 *y, int n,
    Functor functor) {
  int vec_size = std::min(phi::GetVectorizedSize(x), phi::GetVectorizedSize(y));
  auto stream = dev_ctx.stream();
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, vec_size);
  auto block_num = config.block_per_grid;
  auto thread_num = config.thread_per_block;
#define UNARY_FUNCTOR_CASE(__vec_size)                                         \
  VectorizedUnaryCUDAKernel<T1, T2, Functor,                                   \
                            __vec_size><<<block_num, thread_num, 0, stream>>>( \
      x, y, n, functor);
  switch (vec_size) {
    case 8:
      UNARY_FUNCTOR_CASE(8);
      break;
    case 4:
      UNARY_FUNCTOR_CASE(4);
      break;
    case 2:
      UNARY_FUNCTOR_CASE(2);
      break;
    case 1:
      UNARY_FUNCTOR_CASE(1);
      break;
    default:
      PADDLE_THROW("Unsupported vec_size = %d", vec_size);
  }
#undef UNARY_FUNCTOR_CASE
}

template <typename T1, typename T2, typename T3, typename Functor>
static void LaunchVectorizedBinaryFunctorKernel(
    const platform::CUDADeviceContext &dev_ctx, const T1 *x, const T2 *y, T3 *z,
    int n, Functor functor) {
  int vec_size = std::min(phi::GetVectorizedSize(x), phi::GetVectorizedSize(y));
  vec_size = std::min(vec_size, phi::GetVectorizedSize(z));
  auto stream = dev_ctx.stream();
  auto config = platform::GetGpuLaunchConfig1D(dev_ctx, n, vec_size);
  auto block_num = config.block_per_grid;
  auto thread_num = config.thread_per_block;
#define BINARY_FUNCTOR_CASE(__vec_size)                                       \
  VectorizedBinaryCUDAKernel<                                                 \
      T1, T2, T3, Functor, __vec_size><<<block_num, thread_num, 0, stream>>>( \
      x, y, z, n, functor);
  switch (vec_size) {
    case 8:
      BINARY_FUNCTOR_CASE(8);
      break;
    case 4:
      BINARY_FUNCTOR_CASE(4);
      break;
    case 2:
      BINARY_FUNCTOR_CASE(2);
      break;
    case 1:
      BINARY_FUNCTOR_CASE(1);
      break;
    default:
      PADDLE_THROW("Unsupported vec_size = %d", vec_size);
  }
#undef BINARY_FUNCTOR_CASE
}

template <typename T>
struct ReluFwdFunctor {
  HOSTDEVICE T operator()(T x) const { return x * (x > static_cast<T>(0)); }
};

template <>
struct ReluFwdFunctor<platform::float16> {
  HOSTDEVICE platform::float16 operator()(platform::float16 x) const {
    return x > static_cast<platform::float16>(0)
               ? x
               : static_cast<platform::float16>(0);
  }
};

template <typename T>
struct ReluBwdFunctor {
  HOSTDEVICE T operator()(T y, T dy) const {
    return dy * (y > static_cast<T>(0));
  }
};

template <>
struct ReluBwdFunctor<platform::float16> {
  HOSTDEVICE platform::float16 operator()(platform::float16 y,
                                          platform::float16 dy) const {
    return y > static_cast<platform::float16>(0)
               ? dy
               : static_cast<platform::float16>(0);
  }
};

template <typename T>
struct AddReluFwdFunctor {
  HOSTDEVICE T operator()(T x, T y) const { return ReluFwdFunctor<T>()(x + y); }
};

inline __device__ bool IsEqual(float x, float y) {
  return x == y || (isnan(x) && isnan(y));
}

static __global__ void CheckMaskValue(const float *x, const void *mask,
                                      int *cnt, uint32_t n) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    uint8_t m = reinterpret_cast<const uint8_t *>(mask)[idx / 8];
    if (cnt && cnt[idx / 8] != 1) {
      printf("Wrong cnt value 0 %d %d\n", cnt[idx / 8], static_cast<int>(idx));
      asm("trap;");
    }

    uint32_t loc = idx % 8;
    bool flag = ((m & (1 << loc)) != 0);
    if (flag) {
      if (x[idx] < 0) {
        printf("Wrong mask value 1 %f\n", x[idx]);
        asm("trap;");
      }
    } else {
      if (x[idx] > 0) {
        printf("Wrong mask value 2 %f\n", x[idx]);
        asm("trap;");
      }
    }
  }
}

void LaunchVectorized128MaskedReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, float *y,
    void *mask, size_t n);

void LaunchVectorized128MaskedAddReluFwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *x, const float *y,
    float *z, void *mask, size_t n);

void LaunchVectorized128MaskedReluBwdKernel(
    const platform::CUDADeviceContext &dev_ctx, const float *dy,
    const void *mask, float *dx, size_t n);

template <typename T>
static void LaunchReluFwdCUDAKernel(const platform::CUDADeviceContext &dev_ctx,
                                    const T *x, void *mask, T *y, int n) {
  if (std::is_same<T, float>::value) {
    auto *x_ptr = reinterpret_cast<const float *>(x);
    auto *y_ptr = reinterpret_cast<float *>(y);
    void *mask_ptr = mask;
    LaunchVectorized128MaskedReluFwdKernel(dev_ctx, x_ptr, y_ptr, mask_ptr, n);

    /*
    config = platform::GetGpuLaunchConfig1D(dev_ctx, n);
    CheckMaskValue<<<config.block_per_grid, config.thread_per_block, 0,
                     dev_ctx.stream()>>>(x_ptr, mask_ptr, nullptr, n);
    */
    return;
  }
  LaunchVectorizedUnaryFunctorKernel<T, T, ReluFwdFunctor<T>>(
      dev_ctx, x, y, n, ReluFwdFunctor<T>());
}

template <typename T>
static void LaunchAddReluFwdCUDAKernel(
    const platform::CUDADeviceContext &dev_ctx, const T *x, const T *y,
    void *mask, T *z, int n) {
  if (std::is_same<T, float>::value) {
    auto *x_ptr = reinterpret_cast<const float *>(x);
    auto *y_ptr = reinterpret_cast<const float *>(y);
    auto *z_ptr = reinterpret_cast<float *>(z);
    void *mask_ptr = mask;
    LaunchVectorized128MaskedAddReluFwdKernel(dev_ctx, x_ptr, y_ptr, z_ptr,
                                              mask_ptr, n);

    /*
config = platform::GetGpuLaunchConfig1D(dev_ctx, n);
CheckMaskValue<<<config.block_per_grid, config.thread_per_block, 0,
dev_ctx.stream()>>>(x_ptr, mask_ptr, nullptr, n);
*/
    return;
  }
  LaunchVectorizedBinaryFunctorKernel<T, T, T, AddReluFwdFunctor<T>>(
      dev_ctx, x, y, z, n, AddReluFwdFunctor<T>());
}

template <typename T, bool UseFastKernel = false>
static void LaunchReluBwdCUDAKernel(const platform::CUDADeviceContext &dev_ctx,
                                    const T *y, const T *dy, const void *mask,
                                    T *dx, int n) {
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(y) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(dy) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(dx) % 128, 0);
  PADDLE_ENFORCE_EQ(reinterpret_cast<uintptr_t>(mask) % 128, 0);
  if (std::is_same<T, float>::value) {
    auto *dy_ptr = reinterpret_cast<const float *>(dy);
    auto *dx_ptr = reinterpret_cast<float *>(dx);
    const void *mask_ptr = mask;

    /*
    config = platform::GetGpuLaunchConfig1D(dev_ctx, n);
    CheckMaskValue<<<config.block_per_grid, config.thread_per_block, 0,
                     dev_ctx.stream()>>>(y_ptr, mask_ptr, nullptr, n);
    */

    LaunchVectorized128MaskedReluBwdKernel(dev_ctx, dy_ptr, mask_ptr, dx_ptr,
                                           n);
    return;
  }
  LaunchVectorizedBinaryFunctorKernel<T, T, T, ReluBwdFunctor<T>>(
      dev_ctx, y, dy, dx, n, ReluBwdFunctor<T>());
}

template <typename T>
static T *TransformLayout(const platform::CUDADeviceContext &dev_ctx,
                          const framework::Tensor &x, framework::Tensor *y,
                          framework::DataLayout x_layout) {
  cudnnTensorDescriptor_t x_desc, y_desc;
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cudnnCreateTensorDescriptor(&x_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cudnnCreateTensorDescriptor(&y_desc));

  const auto &x_dims = x.dims();

  int N, C, H, W, D;
  ExtractNCWHD(x_dims, x_layout, &N, &C, &H, &W, &D);

  int dims[] = {N, C, H, W, D};
  int strides_nchw[] = {C * H * W * D, H * W * D, W * D, D, 1};
  int strides_nhwc[] = {H * W * D * C, 1, W * D * C, D * C, C};
  int *x_stride, *y_stride;
  if (x_layout == framework::DataLayout::NCHW) {
    y->Resize({N, H, W, C});
    x_stride = strides_nchw;
    y_stride = strides_nhwc;
  } else {
    y->Resize({N, C, H, W});
    x_stride = strides_nhwc;
    y_stride = strides_nchw;
  }

  const auto *x_data = x.template data<T>();
  auto *y_data = y->template mutable_data<T>(dev_ctx.GetPlace());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
      x_desc, platform::CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4, dims, x_stride));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
      y_desc, platform::CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4, dims, y_stride));

  T one = static_cast<T>(1);
  T zero = static_cast<T>(0);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnTransformTensor(
      dev_ctx.cudnn_handle(), &one, x_desc, x_data, &zero, y_desc, y_data));

  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cudnnDestroyTensorDescriptor(x_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cudnnDestroyTensorDescriptor(y_desc));
  return y_data;
}

}  // namespace operators
}  // namespace paddle
