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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/tensor_util.h"

template <typename T>
static __global__ void SimpleElemwiseAddCUDAKernel(const T* x, const T* y, T* z,
                                                   int64_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    z[col] = x[col] + y[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename T>
void EigenSpeed(const paddle::platform::CUDADeviceContext& context,
                const paddle::framework::Tensor& x,
                const paddle::framework::Tensor& y,
                paddle::framework::Tensor* z, int loop = 1000) {
  auto& place = *context.eigen_device();
  for (int i = 0; i < loop; i++) {
    auto eigen_x = paddle::framework::EigenVector<T>::Flatten(x);
    auto eigen_y = paddle::framework::EigenVector<T>::Flatten(y);
    auto eigen_z = paddle::framework::EigenVector<T>::Flatten(*z);
    eigen_z.device(place) = eigen_x + eigen_y;
  }
  context.Wait();
}

template <typename T>
void Body(const paddle::framework::DDim& dims, int loop = 1000) {
  auto cpu_place = paddle::platform::CPUPlace();
  auto gpu_place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceContext context(gpu_place);

  paddle::framework::Tensor in;
  paddle::framework::Tensor x, y, z;
  paddle::framework::Tensor out;

  T* in_ptr = in.mutable_data<T>(dims, cpu_place);
  size_t size = static_cast<size_t>(paddle::framework::product(dims));
  for (int i = 0; i < size; i++) {
    in_ptr[i] = i;
  }

  T* x_ptr = x.mutable_data<T>(dims, gpu_place);
  T* y_ptr = y.mutable_data<T>(dims, gpu_place);
  T* z_ptr = z.mutable_data<T>(dims, gpu_place);

  paddle::framework::TensorCopy(in, gpu_place, context, &x);
  paddle::framework::TensorCopy(in, gpu_place, context, &y);
  paddle::framework::TensorCopy(in, gpu_place, context, &z);
  context.Wait();

  { EigenSpeed<T>(context, x, y, &z, loop); }

  {
    for (int i = 0; i < loop; i++) {
      SimpleElemwiseAddCUDAKernel<
          T><<<(size + 511) / 512, 512, 0, context.stream()>>>(x_ptr, y_ptr,
                                                               z_ptr, size);
    }
    context.Wait();
  }
}

TEST(DataTypeTransform, GPUTransform) {
  paddle::framework::DDim dims({1024, 1024});
  Body<float>(dims, 10000);
  Body<paddle::platform::float16>(dims, 10000);
}
