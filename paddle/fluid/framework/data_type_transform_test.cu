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

#include "paddle/fluid/framework/data_type_transform.h"

#include "gtest/gtest.h"

TEST(DataTypeTransform, GPUTransform) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto cpu_place = CPUPlace();
  auto gpu_place = CUDAPlace(0);
  CUDADeviceContext context(gpu_place);

  auto kernel_fp16 = OpKernelType(proto::VarType::FP16, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_fp32 = OpKernelType(proto::VarType::FP32, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_fp64 = OpKernelType(proto::VarType::FP64, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_int32 = OpKernelType(proto::VarType::INT32, place,
                                   DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_int64 = OpKernelType(proto::VarType::INT64, place,
                                   DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_bool = OpKernelType(proto::VarType::BOOL, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);

  // data type transform from float32
  {
    Tensor in;
    Tensor in_gpu;
    Tensor out_gpu;
    Tensor out;

    float* in_ptr = in.mutable_data<float>(make_ddim({2, 3}), cpu_place);
    float arr[6] = {0, 1, 2, 3, 4, 5};
    int data_number = sizeof(arr) / sizeof(arr[0]);
    memcpy(in_ptr, arr, sizeof(arr));

    TensorCopy(in, gpu_place, context, &in_gpu);

    TransDataType(kernel_fp32, kernel_fp64, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);

    double* out_data_double = out.data<double>();
    context.Wait();
    for (int i = 0; i < data_number; ++i) {
      ASSERT_EQ(out_data_double[i], static_cast<double>(i));
    }

    TransDataType(kernel_fp32, kernel_int32, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);

    int* out_data_int = out.data<int>();
    context.Wait();
    for (int i = 0; i < data_number; ++i) {
      ASSERT_EQ(out_data_int[i], static_cast<int>(i));
    }
  }

  // data type transform from/to float16
}