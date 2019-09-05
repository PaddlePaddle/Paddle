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
    z[col] = static_cast<float>(x[col]) + static_cast<float>(y[col]);
    col += blockDim.x * gridDim.x;
  }
}

template <typename T>
void EigenSpeed(const platform::CUDADeviceContext& context,
                const framework::Tensor& x, const framework::Tensor& y,
                framework::Tensor* z, int loop = 1000) {
  auto& place = *context.eigen_device();
  for (int i = 0; i < loop; i++) {
    auto eigen_x = paddle::framework::EigenVector<T>::Flatten(x);
    auto eigen_y = paddle::framework::EigenVector<T>::Flatten(y);
    auto eigen_z = paddle::framework::EigenVector<T>::Flatten(*z);
    eigen_z.device(place) = eigen_x + eigen_y;
  }
  context.Wait();
}

TEST(DataTypeTransform, GPUTransform) {
  auto cpu_place = paddle::platform::CPUPlace();
  auto gpu_place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceContext context(gpu_place);

  paddle::framework::Tensor in;
  paddle::framework::Tensor x, y, z;
  paddle::framework::Tensor out;

  paddle::framework::DDim dims({1024, 1024});
  float* in_ptr = in.mutable_data<float>(dims, cpu_place);
  for (int i = 0; i < 1024 * 1024; i++) {
    in_ptr[i] = i;
  }

  float* x_ptr = x.mutable_data<float>(dims, gpu_place);
  float* y_ptr = y.mutable_data<float>(dims, gpu_place);
  float* z_ptr = z.mutable_data<float>(dims, gpu_place);

  paddle::framework::TensorCopy(in, gpu_place, context, &x);
  paddle::framework::TensorCopy(in, gpu_place, context, &y);
  paddle::framework::TensorCopy(in, gpu_place, context, &z);
  context.Wait();

  {
    EigenSpeed<float>(context, x, y, &z);
    EigenSpeed<platform::float16>(context, x, y, &z);
  }

  {
    size_t size = static_cast<size_t>(paddle::framework::product(dims));
    for (int i = 0; i < 1000; i++) {
      SimpleElemwiseAddCUDAKernel<
          float><<<(size + 255) / 256, 256, 0, context.stream()>>>(x_ptr, y_ptr,
                                                                   z_ptr, size);
    }
    context.Wait();
  }
}
