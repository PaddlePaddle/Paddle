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

TEST(DataTypeTransform, CPUTransform) {
  auto place = paddle::platform::CPUPlace();

  auto kernel_fp16 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::FP16, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_bf16 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::BF16, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_fp32 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::FP32, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_fp64 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::FP64, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_int32 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::INT32, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_int64 = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::INT64, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  auto kernel_bool = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::BOOL, place,
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain);

  // data type transform from float32
  {
    paddle::framework::Tensor in;
    paddle::framework::Tensor out;

    float* ptr = in.mutable_data<float>(phi::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i / 3;
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(i / 3));
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(i / 3));
    }
  }

  // data type transform from/to float16
  {
    paddle::framework::Tensor in;
    paddle::framework::Tensor out;

    paddle::platform::float16* ptr = in.mutable_data<paddle::platform::float16>(
        phi::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from float16 to other data types
    paddle::framework::TransDataType(kernel_fp16, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_fp16, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to float16
    float* in_data_float =
        in.mutable_data<float>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_fp16, in, &out);
    ptr = out.data<paddle::platform::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::float16>(in_data_float[i]).x);
    }

    // transform double to float16
    double* in_data_double =
        in.mutable_data<double>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_fp16, in, &out);
    ptr = out.data<paddle::platform::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::float16>(in_data_double[i]).x);
    }

    // transform int to float16
    int* in_data_int = in.mutable_data<int>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    paddle::framework::TransDataType(kernel_int32, kernel_fp16, in, &out);
    ptr = out.data<paddle::platform::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::float16>(in_data_int[i]).x);
    }

    // transform int64 to float16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_fp16, in, &out);
    ptr = out.data<paddle::platform::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::float16>(in_data_int64[i]).x);
    }

    // transform bool to float16
    bool* in_data_bool = in.mutable_data<bool>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_fp16, in, &out);
    ptr = out.data<paddle::platform::float16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::float16>(in_data_bool[i]).x);
    }
  }

  // data type transform from/to bfloat16
  {
    paddle::framework::Tensor in;
    paddle::framework::Tensor out;

    paddle::platform::bfloat16* ptr =
        in.mutable_data<paddle::platform::bfloat16>(phi::make_ddim({2, 3}),
                                                    place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from bfloat16 to other data types
    paddle::framework::TransDataType(kernel_bf16, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to bfloat16
    float* in_data_float =
        in.mutable_data<float>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_bf16, in, &out);
    ptr = out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::bfloat16>(in_data_float[i]).x);
    }

    // transform double to bfloat16
    double* in_data_double =
        in.mutable_data<double>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_bf16, in, &out);
    ptr = out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::bfloat16>(in_data_double[i]).x);
    }

    // transform int to bfloat16
    int* in_data_int = in.mutable_data<int>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int[i] = i;
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bf16, in, &out);
    ptr = out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::bfloat16>(in_data_int[i]).x);
    }

    // transform int64 to bfloat16
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_bf16, in, &out);
    ptr = out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::bfloat16>(in_data_int64[i]).x);
    }

    // transform bool to bfloat16
    bool* in_data_bool = in.mutable_data<bool>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_bf16, in, &out);
    ptr = out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i].x,
                static_cast<paddle::platform::bfloat16>(in_data_bool[i]).x);
    }
  }

  // data type transform from/to int32
  {
    paddle::framework::Tensor in;
    paddle::framework::Tensor out;

    int32_t* ptr = in.mutable_data<int32_t>(phi::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i;
    }

    // transform from int32 to other data types
    paddle::framework::TransDataType(kernel_int32, kernel_fp32, in, &out);
    float* out_data_float = out.data<float>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_float[i], static_cast<float>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bf16, in, &out);
    paddle::platform::bfloat16* out_data_bf16 =
        out.data<paddle::platform::bfloat16>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bf16[i],
                static_cast<paddle::platform::bfloat16>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_int64, in, &out);
    int64_t* out_data_int64 = out.data<int64_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int64[i], static_cast<int64_t>(ptr[i]));
    }

    paddle::framework::TransDataType(kernel_int32, kernel_bool, in, &out);
    bool* out_data_bool = out.data<bool>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_bool[i], static_cast<bool>(ptr[i]));
    }

    // transform float to int32
    float* in_data_float =
        in.mutable_data<float>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_float[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp32, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_float[i]));
    }

    // transform double to int32
    double* in_data_double =
        in.mutable_data<double>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_double[i] = i;
    }

    paddle::framework::TransDataType(kernel_fp64, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_double[i]));
    }

    // transform bfloat16 to int32
    paddle::platform::bfloat16* in_data_bf16 =
        in.mutable_data<paddle::platform::bfloat16>(phi::make_ddim({2, 3}),
                                                    place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bf16[i] = i;
    }

    paddle::framework::TransDataType(kernel_bf16, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_bf16[i]));
    }

    // transform int64 to int32
    int64_t* in_data_int64 =
        in.mutable_data<int64_t>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_int64[i] = i;
    }

    paddle::framework::TransDataType(kernel_int64, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_int64[i]));
    }

    // transform bool to int32
    bool* in_data_bool = in.mutable_data<bool>(phi::make_ddim({2, 3}), place);
    for (int i = 0; i < data_number; ++i) {
      in_data_bool[i] = i;
    }

    paddle::framework::TransDataType(kernel_bool, kernel_int32, in, &out);
    ptr = out.data<int32_t>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(ptr[i], static_cast<int32_t>(in_data_bool[i]));
    }
  }
}
