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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"

namespace phi {
namespace funcs {

// The following part of the code refers to NVIDIA-cutlass
// https://github.com/NVIDIA/cutlass/blob/master/tools/util/include/cutlass/util/device_nchw_to_nhwc.h
// Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: BSD-3-Clause
template <typename T>
__global__ void batch_transpose_kernel(
    T* output, const T* input, const int batch, const int M, const int N) {
  const int num = M * N;
  // "+1" to avoid smem bank conflict
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid = tid / 32;
  const int32_t lid = tid % 32;
  const int32_t batch_i = blockIdx.z;
  const int32_t mi0 = blockIdx.y * 32;
  const int32_t ni0 = blockIdx.x * 32;

  const size_t input_idx = batch_i * num + (mi0 + wid) * N + ni0;
  const T* A = input + input_idx;
  if (ni0 + lid < N) {
    const int lid_x_33 = lid * 33;
    if ((mi0 + 32) <= M) {
      int mi = wid;  // between 0 and 7
#pragma unroll
      for (int mLoopIdx = 0; mLoopIdx < 4; mLoopIdx++) {
        shbuf[lid_x_33 + mi] = A[lid];
        A = &A[8 * N];
        mi += 8;
      }
    } else {
      for (int mi = wid; mi < 32; mi += 8) {
        if ((mi + mi0) < M) {
          shbuf[lid_x_33 + mi] = A[lid];
        }
        A = &A[8 * N];
      }
    }
  }
  __syncthreads();

  const int32_t miOut = mi0 + lid;
  output = &output[batch_i * num + miOut];
  if (miOut < M) {
    if (ni0 + 32 < N) {
      int nI = wid;
#pragma unroll
      for (int nLoopIdx = 0; nLoopIdx < 4; ++nLoopIdx) {
        output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
        nI += 8;
      }
    } else {
      for (int nI = wid; nI < 32; nI += 8) {
        if (ni0 + nI < N) {
          output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
        }
      }
    }
  }
}

template <typename T>
void BatchTranspose(T* output, const T* input, int batch, int m, int n) {
  dim3 grid((n + 31) / 32, (m + 31) / 32, batch);
  dim3 block(32, 8);
  batch_transpose_kernel<<<grid, block>>>(output, input, batch, m, n);
}

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;

template void BatchTranspose(
    float16* output, const float16* input, int batch, int m, int n);
template void BatchTranspose(
    float* output, const float* input, int batch, int m, int n);

template struct SetConstant<phi::GPUContext, float16>;
template struct SetConstant<phi::GPUContext, bfloat16>;
template struct SetConstant<phi::GPUContext, float>;
template struct SetConstant<phi::GPUContext, double>;
template struct SetConstant<phi::GPUContext, uint8_t>;
template struct SetConstant<phi::GPUContext, int>;
template struct SetConstant<phi::GPUContext, int16_t>;
template struct SetConstant<phi::GPUContext, int64_t>;
template struct SetConstant<phi::GPUContext, bool>;
template struct SetConstant<phi::GPUContext, phi::dtype::complex<float>>;
template struct SetConstant<phi::GPUContext, phi::dtype::complex<double>>;

template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, float16>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext,
                            bfloat16>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, float>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, double>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, uint8_t>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, int>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, int16_t>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, int64_t>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext, bool>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext,
                            phi::dtype::complex<float>>;
template struct SetConstant<paddle::platform::CUDAPinnedDeviceContext,
                            phi::dtype::complex<double>>;

#define DEFINE_GPU_TRANS(RANK)                                     \
  template struct Transpose<phi::GPUContext, bool, RANK>;          \
  template struct Transpose<phi::GPUContext, unsigned char, RANK>; \
  template struct Transpose<phi::GPUContext, float, RANK>;         \
  template struct Transpose<phi::GPUContext, double, RANK>;        \
  template struct Transpose<phi::GPUContext, float16, RANK>;       \
  template struct Transpose<phi::GPUContext, bfloat16, RANK>;      \
  template struct Transpose<phi::GPUContext, int8_t, RANK>;        \
  template struct Transpose<phi::GPUContext, int16_t, RANK>;       \
  template struct Transpose<phi::GPUContext, int32_t, RANK>;       \
  template struct Transpose<phi::GPUContext, int64_t, RANK>;       \
  template struct Transpose<phi::GPUContext,                       \
                            phi::dtype::complex<float>,            \
                            RANK>;                                 \
  template struct Transpose<phi::GPUContext, phi::dtype::complex<double>, RANK>;

DEFINE_GPU_TRANS(1);
DEFINE_GPU_TRANS(2);
DEFINE_GPU_TRANS(3);
DEFINE_GPU_TRANS(4);
DEFINE_GPU_TRANS(5);
DEFINE_GPU_TRANS(6);

#define REINTERPRET(T, DST_PTR, SRC_PTR) \
  T* DST_PTR = reinterpret_cast<T*>(SRC_PTR)

template <typename T>
__global__ void TransposeNormalKernel(const T* in_ptr,
                                      T* out_ptr,
                                      int64_t element,
                                      const int64_t* in_stride_ptr,
                                      const int64_t* out_stride_ptr,
                                      const int64_t* axis_ptr,
                                      int rank) {
  CUDA_KERNEL_LOOP(out_idx, element) {
    int64_t in_idx = 0;
    int64_t tmp_idx = out_idx;
    for (int i = 0; i < rank; ++i) {
      const int64_t coordinate = tmp_idx / out_stride_ptr[i];
      tmp_idx -= coordinate * out_stride_ptr[i];
      in_idx += coordinate * in_stride_ptr[axis_ptr[i]];
    }
    out_ptr[out_idx] = in_ptr[in_idx];
  }
}

template <typename DeviceContext, typename T>
void TransposeNormal<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
    const std::vector<int>& axis) {
  const int rank = axis.size();
  auto in_stride = phi::stride(in.dims());
  auto out_stride = phi::stride(out->dims());
  auto* in_ptr = in.data<T>();
  auto* out_ptr = out->data<T>();

  // copy in_stride, out_stride, axis to gpu device
  const paddle::platform::CUDAPlace& cuda_place = context.GetPlace();
  paddle::platform::CPUPlace cpu_place = paddle::platform::CPUPlace();
  size_t size = 3 * rank * sizeof(int64_t);
  auto cpu_buf_holder = paddle::memory::Alloc(cpu_place, size);
  auto cuda_buf_holder = paddle::memory::Alloc(cuda_place, size);
  REINTERPRET(int64_t, cpu_buf, cpu_buf_holder->ptr());
  REINTERPRET(int64_t, cuda_buf, cuda_buf_holder->ptr());
  for (int i = 0; i < rank; ++i) {
    cpu_buf[i] = in_stride[i];
    cpu_buf[rank + i] = out_stride[i];
    cpu_buf[2 * rank + i] = axis[i];
  }
  paddle::memory::Copy(
      cuda_place, cuda_buf, cpu_place, cpu_buf, size, context.stream());
  REINTERPRET(const int64_t, in_stride_ptr, cuda_buf);
  REINTERPRET(const int64_t, out_stride_ptr, cuda_buf + rank);
  REINTERPRET(const int64_t, axis_ptr, cuda_buf + 2 * rank);

  const int MAX_BLOCK_DIM = context.GetMaxThreadsPerBlock();
  const int MAX_GRID_DIM = context.GetMaxPhysicalThreadCount() / MAX_BLOCK_DIM;
  int64_t elements = in.numel();
  int block_size = (elements >= MAX_BLOCK_DIM)
                       ? MAX_BLOCK_DIM
                       : (1 << static_cast<int>(std::log2(elements)));
  int grid_size = elements / block_size;
  grid_size = (grid_size >= MAX_GRID_DIM) ? MAX_GRID_DIM : grid_size;
  TransposeNormalKernel<T><<<grid_size, block_size, 0, context.stream()>>>(
      in_ptr, out_ptr, elements, in_stride_ptr, out_stride_ptr, axis_ptr, rank);
}

template <typename T>
struct TransposeNormal<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = stride(in.dims());
    auto out_stride = stride(out->dims());
    auto* in_ptr = in.data<T>();
    auto* out_ptr = out->data<T>();

    // copy in_stride, out_stride, axis to gpu device
    const phi::GPUPlace& cuda_place = context.GetPlace();
    phi::CPUPlace cpu_place = paddle::platform::CPUPlace();
    size_t size = 3 * rank * sizeof(int64_t);
    auto cpu_buf_holder = paddle::memory::Alloc(cpu_place, size);
    auto cuda_buf_holder = paddle::memory::Alloc(cuda_place, size);
    REINTERPRET(int64_t, cpu_buf, cpu_buf_holder->ptr());
    REINTERPRET(int64_t, cuda_buf, cuda_buf_holder->ptr());
    for (int i = 0; i < rank; ++i) {
      cpu_buf[i] = in_stride[i];
      cpu_buf[rank + i] = out_stride[i];
      cpu_buf[2 * rank + i] = axis[i];
    }
    paddle::memory::Copy(
        cuda_place, cuda_buf, cpu_place, cpu_buf, size, context.stream());
    REINTERPRET(const int64_t, in_stride_ptr, cuda_buf);
    REINTERPRET(const int64_t, out_stride_ptr, cuda_buf + rank);
    REINTERPRET(const int64_t, axis_ptr, cuda_buf + 2 * rank);

    const int MAX_BLOCK_DIM = context.GetMaxThreadsPerBlock();
    const int MAX_GRID_DIM =
        context.GetMaxPhysicalThreadCount() / MAX_BLOCK_DIM;
    int64_t elements = in.numel();
    int block_size = (elements >= MAX_BLOCK_DIM)
                         ? MAX_BLOCK_DIM
                         : (1 << static_cast<int>(std::log2(elements)));
    int grid_size = elements / block_size;
    grid_size = (grid_size >= MAX_GRID_DIM) ? MAX_GRID_DIM : grid_size;
    TransposeNormalKernel<T>
        <<<grid_size, block_size, 0, context.stream()>>>(in_ptr,
                                                         out_ptr,
                                                         elements,
                                                         in_stride_ptr,
                                                         out_stride_ptr,
                                                         axis_ptr,
                                                         rank);
  }
};

// define transpose normal
#define DEFINE_GPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<phi::GPUContext, TYPE>

DEFINE_GPU_TRANS_NORMAL(float16);
DEFINE_GPU_TRANS_NORMAL(bfloat16);
DEFINE_GPU_TRANS_NORMAL(float);
DEFINE_GPU_TRANS_NORMAL(double);
DEFINE_GPU_TRANS_NORMAL(int);
DEFINE_GPU_TRANS_NORMAL(int64_t);
DEFINE_GPU_TRANS_NORMAL(bool);
DEFINE_GPU_TRANS_NORMAL(int16_t);
DEFINE_GPU_TRANS_NORMAL(uint8_t);
DEFINE_GPU_TRANS_NORMAL(int8_t);
DEFINE_GPU_TRANS_NORMAL(phi::dtype::complex<float>);
DEFINE_GPU_TRANS_NORMAL(phi::dtype::complex<double>);

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const paddle::platform::DeviceContext& context,
                       phi::DenseTensor* tensor,
                       float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    SetConstant<phi::GPUContext, T> functor;
    functor(reinterpret_cast<const phi::GPUContext&>(context_),
            tensor_,
            static_cast<T>(value_));
  }

  const paddle::platform::DeviceContext& context_;
  phi::DenseTensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<paddle::platform::CUDAPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    float value) {
  phi::VisitDataType(tensor->dtype(),
                     TensorSetConstantGPU(context, tensor, value));
}

template <typename T>
__global__ void RowwiseAddKernel(
    const T* a, const T* b, T* c, int width, int num) {
  T tmp = 1.0 / width;
  CUDA_KERNEL_LOOP(i, num) {
    int h = i * tmp;
    int w = i - h * width;
    c[i] = a[i] + b[w];
  }
}

template <typename T>
struct RowwiseAdd<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& vector,
                  phi::DenseTensor* output) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(
        vector.numel(),
        size,
        phi::errors::InvalidArgument(
            "The input vector size"
            " should be equal to the size of each row of input tensor."
            " Expected vector size=%d, but received %d",
            size,
            vector.numel()));
    const char* in_dims_cstr = in_dims.to_str().c_str();
    const char* out_dims_cstr = out_dims.to_str().c_str();
    PADDLE_ENFORCE_EQ(
        out_dims,
        in_dims,
        phi::errors::InvalidArgument(
            "The output tensor shape should be same as the input tensor"
            " shape. Expected output tensor shape: %s,"
            " but received %s",
            in_dims_cstr,
            out_dims_cstr));
    int blocks = 512;
    int grids = (input.numel() + blocks - 1) / blocks;
    RowwiseAddKernel<T><<<grids, blocks, 0, context.stream()>>>(
        input.data<T>(),
        vector.data<T>(),
        output->data<T>(),
        static_cast<int>(in_dims[1]),
        static_cast<int>(input.numel()));
  }
};

template struct RowwiseAdd<phi::GPUContext, float>;
template struct RowwiseAdd<phi::GPUContext, double>;
template struct ColwiseSum<phi::GPUContext, float>;
template struct ColwiseSum<phi::GPUContext, int>;
template struct ColwiseSum<phi::GPUContext, int64_t>;
// template struct ColwiseSum<phi::GPUContext, double>;
// The ColwiseSum<phi::GPUContext, double> failed in debug
// mode,
// and only failed for this case. So reimplemented it.
template <>
void ColwiseSum<phi::GPUContext, double>::operator()(
    const phi::GPUContext& context,
    const phi::DenseTensor& input,
    phi::DenseTensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(),
                    size,
                    phi::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor column"
                        " dimension. Expected vector size=%d, but received %d",
                        size,
                        vector->numel()));
  phi::DenseTensor one;
  one.Resize({in_dims[0]});
  context.template Alloc<double>(&one);

  SetConstant<phi::GPUContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  phi::funcs::GetBlas<phi::GPUContext, double>(context).GEMV(
      true,
      static_cast<int>(in_dims[0]),
      static_cast<int>(in_dims[1]),
      1.0,
      input.data<double>(),
      one.data<double>(),
      0.0,
      vector->data<double>());
}

template struct RowwiseSum<phi::GPUContext, float>;
// template struct RowwiseSum<phi::GPUContext, double>;
// TODO(zcd): Following ColwiseSum format, need to confirm.
// The RowwiseSum<phi::GPUContext, double> failed in debug
// mode,
// and only failed for this case. So reimplemented it.
template <>
void RowwiseSum<phi::GPUContext, double>::operator()(
    const phi::GPUContext& context,
    const phi::DenseTensor& input,
    phi::DenseTensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(),
                    in_dims[0],
                    phi::errors::InvalidArgument(
                        "The size of input vector"
                        " should be equal to the size of input tensor row"
                        " dimension. Expected vector size=%d, but received %d",
                        in_dims[0],
                        vector->numel()));
  phi::DenseTensor one;
  one.Resize({size});
  context.template Alloc<double>(&one);

  SetConstant<phi::GPUContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  phi::funcs::GetBlas<phi::GPUContext, double>(context).GEMV(
      true,
      static_cast<int>(in_dims[1]),
      static_cast<int>(in_dims[0]),
      1.0,
      one.data<double>(),
      input.data<double>(),
      0.0,
      vector->data<double>());
}

template struct RowwiseMean<phi::GPUContext, float>;
template struct RowwiseMean<phi::GPUContext, double>;

}  // namespace funcs
}  // namespace phi
