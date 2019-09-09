/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_fp16.h>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/float16.h"

template <typename T>
static __global__ void SimpleElemwiseAddCUDAKernel(const T* x, const T* y, T* z,
                                                   int64_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    z[col] = x[col] + y[col];
    col += blockDim.x * gridDim.x;
  }
}

__global__ void halfadd(const half* x, const half* y, half* z, int64_t n) {
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  int n2 = n / 2;
  half2* x2 = reinterpret_cast<half2*>(x);
  half2* y2 = reinterpret_cast<half2*>(y);
  half2* z2 = reinterpret_cast<half2*>(z);

  for (int i = start; i < n2; i += stride) z2[i] = __hadd2(x2[i], y2[i]);

  // first thread handles singleton for odd arrays
  if (start == 0 && (n % 2)) z[n - 1] = x[n - 1] + y[n - 1];
}

/*
__global__ void halfadd(const half2 *x, const half2 *y, half2* z, int64_t size)
{
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < size; i+= stride)
        z[i] = __hadd2(x[i], y[i]);
}
*/

template <typename T>
void SetTensor(const paddle::framework::DDim& dims,
               paddle::framework::Tensor* x, paddle::framework::Tensor* y,
               paddle::framework::Tensor* z,
               const paddle::platform::CUDADeviceContext& context) {
  auto cpu_place = paddle::platform::CPUPlace();
  auto gpu_place = paddle::platform::CUDAPlace(0);

  paddle::framework::Tensor in;

  T* in_ptr = in.mutable_data<T>(dims, cpu_place);
  size_t size = static_cast<size_t>(paddle::framework::product(dims));
  for (int i = 0; i < size; i++) {
    in_ptr[i] = i / 1000;
  }

  paddle::framework::TensorCopy(in, gpu_place, context, x);
  paddle::framework::TensorCopy(in, gpu_place, context, y);
  paddle::framework::TensorCopy(in, gpu_place, context, z);
  context.Wait();
}

template <typename T>
void EigenSpeed(const paddle::framework::DDim& dims, int loop = 1000) {
  paddle::framework::Tensor x, y, z;
  auto gpu_place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  SetTensor<T>(dims, &x, &y, &z, context);

  auto& place = *context.eigen_device();
  for (int i = 0; i < loop; i++) {
    auto eigen_x = paddle::framework::EigenVector<T>::Flatten(x);
    auto eigen_y = paddle::framework::EigenVector<T>::Flatten(y);
    auto eigen_z = paddle::framework::EigenVector<T>::Flatten(z);
    eigen_z.device(place) = eigen_x + eigen_y;
  }
  context.Wait();
}

template <typename T>
void SimpleSpeed(const paddle::framework::DDim& dims, int loop = 1000) {
  paddle::framework::Tensor x, y, z;
  auto gpu_place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  SetTensor<T>(dims, &x, &y, &z, context);

  T* x_ptr = x.mutable_data<T>(dims, gpu_place);
  T* y_ptr = y.mutable_data<T>(dims, gpu_place);
  T* z_ptr = z.mutable_data<T>(dims, gpu_place);
  size_t size = static_cast<size_t>(paddle::framework::product(dims));

  for (int i = 0; i < loop; i++) {
    SimpleElemwiseAddCUDAKernel<
        T><<<(size + 511) / 512, 512, 0, context.stream()>>>(x_ptr, y_ptr,
                                                             z_ptr, size);
  }
  context.Wait();
}

template <typename T>
void Half2Speed(const paddle::framework::DDim& dims, int loop = 1000) {
  paddle::framework::Tensor x, y, z;
  auto gpu_place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  SetTensor<T>(dims, &x, &y, &z, context);

  T* x_ptr = x.mutable_data<T>(dims, gpu_place);
  T* y_ptr = y.mutable_data<T>(dims, gpu_place);
  T* z_ptr = z.mutable_data<T>(dims, gpu_place);

  half* x2 = (reinterpret_cast<half*>(x_ptr));
  half* y2 = (reinterpret_cast<half*>(y_ptr));
  half* z2 = (reinterpret_cast<half*>(z_ptr));
  size_t size = static_cast<size_t>(paddle::framework::product(dims));

  int64_t size2 = size / 2;
  for (int i = 0; i < loop; i++) {
    halfadd<<<(size2 + 511) / 512, 512, 0, context.stream()>>>(x2, y2, z2,
                                                               size);
  }
  context.Wait();
}

template <typename T>
void TestSpeed(const paddle::framework::DDim& dims, int loop = 1000) {
  EigenSpeed<T>(dims, loop);
  SimpleSpeed<T>(dims, loop);
  Half2Speed<T>(dims, loop);
}

TEST(DataTypeTransform, GPUTransform) {
  paddle::framework::DDim dims({1024, 1024});
  // TestSpeed<float>(dims, 1000);
  TestSpeed<paddle::platform::float16>(dims, 10000);
}
