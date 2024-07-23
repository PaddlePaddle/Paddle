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
#include "paddle/fluid/framework/tensor_util.h"

TEST(DataTypeTransform, GPUTransform) {
  auto cpu_place = phi::CPUPlace();
  auto gpu_place = phi::GPUPlace(0);
  phi::GPUContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  auto kernel_fp16 = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT16);

  auto kernel_fp32 = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);

  auto kernel_fp64 = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT64);

  auto kernel_int32 = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT32);

  auto kernel_int64 = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::INT64);

  auto kernel_bool = phi::KernelKey(
      gpu_place, phi::DataLayout::ALL_LAYOUT, phi::DataType::BOOL);

  // data type transform from float32
  {
    phi::DenseTensor in;
    phi::DenseTensor in_gpu;
    phi::DenseTensor out_gpu;
    phi::DenseTensor out;

    float* in_ptr =
        in.mutable_data<float>(common::make_ddim({2, 3}), cpu_place);
    float arr[6] = {0, 1, 2, 3, 4, 5};
    int data_number = sizeof(arr) / sizeof(arr[0]);
    memcpy(in_ptr, arr, sizeof(arr));

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_fp32, kernel_fp64, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(arr[i]));
    }

    paddle::framework::TransDataType(
        kernel_fp32, kernel_int32, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(arr[i]));
    }
  }

  // data type transform from/to float16
  {
    phi::DenseTensor in;
    phi::DenseTensor in_gpu;
    phi::DenseTensor out_gpu;
    phi::DenseTensor out;

    phi::dtype::float16* ptr = in.mutable_data<phi::dtype::float16>(
        common::make_ddim({2, 3}), cpu_place);
    phi::dtype::float16 arr[6] = {phi::dtype::float16(0),
                                  phi::dtype::float16(1),
                                  phi::dtype::float16(2),
                                  phi::dtype::float16(3),
                                  phi::dtype::float16(4),
                                  phi::dtype::float16(5)};

    int data_number = sizeof(arr) / sizeof(arr[0]);
    memcpy(ptr, arr, sizeof(arr));
    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();

    // transform from float16 to other data types
    paddle::framework::TransDataType(
        kernel_fp16, kernel_fp32, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(
        kernel_fp16, kernel_fp64, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(
        kernel_fp16, kernel_int32, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    paddle::framework::TransDataType(
        kernel_fp16, kernel_int64, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(
        kernel_fp16, kernel_bool, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to float16
    float* in_data_float =
        in.mutable_data<float>(common::make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = i;
    }

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_fp32, kernel_fp16, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_float[i]).x);
    }

    // transform double to float16
    double* in_data_double =
        in.mutable_data<double>(common::make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_fp64, kernel_fp16, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<phi::dtype::float16>(in_data_double[i]).x);
    }

    // transform int to float16
    int* in_data_int =
        in.mutable_data<int>(common::make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_int32, kernel_fp16, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_int[i]).x);
    }

    // transform int64 to float16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(common::make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_int64, kernel_fp16, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_int64[i]).x);
    }

    // transform bool to float16
    bool* in_data_bool =
        in.mutable_data<bool>(common::make_ddim({2, 3}), cpu_place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::framework::TransDataType(
        kernel_bool, kernel_fp16, in_gpu, &out_gpu);
    paddle::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    ptr = out.data<phi::dtype::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x, static_cast<phi::dtype::float16>(in_data_bool[i]).x);
    }
  }
}
