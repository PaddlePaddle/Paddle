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
#include "paddle/fluid/framework/tensor_util.h"

#include "gtest/gtest.h"

TEST(DataTypeTransform, GPUTransform) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto cpu_place = CPUPlace();
  auto gpu_place = CUDAPlace(0);
  CUDADeviceContext context(gpu_place);

  auto kernel_fp16 = OpKernelType(proto::VarType::FP16, gpu_place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_fp32 = OpKernelType(proto::VarType::FP32, gpu_place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_fp64 = OpKernelType(proto::VarType::FP64, gpu_place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_int32 = OpKernelType(proto::VarType::INT32, gpu_place,
                                   DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_int64 = OpKernelType(proto::VarType::INT64, gpu_place,
                                   DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_bool = OpKernelType(proto::VarType::BOOL, gpu_place,
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
    context.Wait();
    TransDataType(kernel_fp32, kernel_fp64, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(arr[i]));
    }

    TransDataType(kernel_fp32, kernel_int32, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(arr[i]));
    }
  }

  // data type transform from/to float16
  {
    Tensor in;
    Tensor in_gpu;
    Tensor out_gpu;
    Tensor out;

    float16* ptr = in.mutable_data<float16>(make_ddim({2, 3}), cpu_place);
    float16 arr[6] = {float16(0), float16(1), float16(2),
                      float16(3), float16(4), float16(5)};
    int data_number = sizeof(arr) / sizeof(arr[0]);
    memcpy(ptr, arr, sizeof(arr));
    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();

    // transform from float16 to other data types
    TransDataType(kernel_fp16, kernel_fp32, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    TransDataType(kernel_fp16, kernel_fp64, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    TransDataType(kernel_fp16, kernel_int32, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    TransDataType(kernel_fp16, kernel_int64, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    TransDataType(kernel_fp16, kernel_bool, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to float16
    float* in_data_float = in.mutable_data<float>(make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = i;
    }

    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    TransDataType(kernel_fp32, kernel_fp16, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<float16>(in_data_float[i]).x);
    }

    // transform double to float16
    double* in_data_double =
        in.mutable_data<double>(make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    TransDataType(kernel_fp64, kernel_fp16, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<float16>(in_data_double[i]).x);
    }

    // transform int to float16
    int* in_data_int = in.mutable_data<int>(make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    TransDataType(kernel_int32, kernel_fp16, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<float16>(in_data_int[i]).x);
    }

    // transform int64 to float16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    TransDataType(kernel_int64, kernel_fp16, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<float16>(in_data_int64[i]).x);
    }

    // transform bool to float16
    bool* in_data_bool = in.mutable_data<bool>(make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    TransDataType(kernel_bool, kernel_fp16, in_gpu, &out_gpu);
    TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<float16>(in_data_bool[i]).x);
    }
  }
}
