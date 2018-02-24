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
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto place = CPUPlace();

  Tensor in;
  Tensor out;

  float* ptr = in.mutable_data<float>(make_ddim({2, 3}), place);
  int data_number = 2 * 3;

  for (int i = 0; i < data_number; ++i) {
    ptr[i] = i / 3;
  }

  auto kernel_fp32 = OpKernelType(proto::VarType::FP32, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_fp64 = OpKernelType(proto::VarType::FP64, place,
                                  DataLayout::kAnyLayout, LibraryType::kPlain);
  auto kernel_int32 = OpKernelType(proto::VarType::INT32, place,
                                   DataLayout::kAnyLayout, LibraryType::kPlain);

  TransDataType(kernel_fp32, kernel_fp64, in, &out);
  double* out_data_double = out.data<double>();
  for (int i = 0; i < data_number; ++i) {
    ASSERT_EQ(out_data_double[i], static_cast<double>(i / 3));
  }

  TransDataType(kernel_fp32, kernel_int32, in, &out);
  int* out_data_int = out.data<int>();
  for (int i = 0; i < data_number; ++i) {
    ASSERT_EQ(out_data_int[i], static_cast<int>(i / 3));
  }
}
