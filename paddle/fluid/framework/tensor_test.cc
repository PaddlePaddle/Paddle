//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/tensor.h"
#include <gtest/gtest.h>
#include <string>

namespace framework = paddle::framework;
namespace platform = paddle::platform;

TEST(Tensor, Dims) {
  framework::Tensor tt;
  tt.Resize({2, 3, 4});
  framework::DDim dims = tt.dims();
  ASSERT_EQ(arity(dims), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(i + 2, dims[i]);
  }
}

TEST(Tensor, DataAssert) {
  framework::Tensor src_tensor;

  bool caught = false;
  try {
    src_tensor.data<double>();
  } catch (platform::EnforceNotMet err) {
    caught = true;
    std::string msg =
        "holder_ should not be null\nTensor holds no memory. Call "
        "Tensor::mutable_data first.";
    const char* what = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

TEST(Tensor, MutableData) {
  {
    framework::Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({1, 2, 3}),
                                        platform::CPUPlace());
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({3, 4}),
                                        platform::CPUPlace());
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2, 3}),
                                        platform::CPUPlace());
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2}),
                                        platform::CPUPlace());
    EXPECT_EQ(p1, p2);
  }

#ifdef PADDLE_WITH_CUDA
  {
    framework::Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({1, 2, 3}),
                                        platform::CUDAPlace());
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({3, 4}),
                                        platform::CUDAPlace());
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2, 3}),
                                        platform::CUDAPlace());
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2}),
                                        platform::CUDAPlace());
    EXPECT_EQ(p1, p2);
  }
#endif
}

TEST(Tensor, ShareDataWith) {
  {
    framework::Tensor src_tensor;
    framework::Tensor dst_tensor;
    // Try to share data form uninitialized tensor
    bool caught = false;
    try {
      dst_tensor.ShareDataWith(src_tensor);
    } catch (paddle::platform::EnforceNotMet err) {
      caught = true;
      std::string msg =
          "holder_ should not be null\nTensor holds no memory. Call "
          "Tensor::mutable_data first.";
      const char* what = err.what();
      for (size_t i = 0; i < msg.length(); ++i) {
        ASSERT_EQ(what[i], msg[i]);
      }
    }
    ASSERT_TRUE(caught);

    src_tensor.mutable_data<int>(framework::make_ddim({2, 3, 4}),
                                 platform::CPUPlace());
    dst_tensor.ShareDataWith(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }

#ifdef PADDLE_WITH_CUDA
  {
    framework::Tensor src_tensor;
    framework::Tensor dst_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({2, 3, 4}),
                                 platform::CUDAPlace());
    dst_tensor.ShareDataWith(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }
#endif
}

TEST(Tensor, Slice) {
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({5, 3, 4}),
                                 platform::CPUPlace());
    framework::Tensor slice_tensor = src_tensor.Slice(1, 3);
    framework::DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 3);
    EXPECT_EQ(slice_dims[0], 2);
    EXPECT_EQ(slice_dims[1], 3);
    EXPECT_EQ(slice_dims[2], 4);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<int>());
    uintptr_t src_mutable_data_address = reinterpret_cast<uintptr_t>(
        src_tensor.mutable_data<int>(src_tensor.dims(), platform::CPUPlace()));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<int>());
    uintptr_t slice_mutable_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.mutable_data<int>(
            slice_tensor.dims(), platform::CPUPlace()));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    EXPECT_EQ(slice_data_address, slice_mutable_data_address);
    EXPECT_EQ(src_data_address + 3 * 4 * 1 * sizeof(int), slice_data_address);
  }

#ifdef PADDLE_WITH_CUDA
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<double>(framework::make_ddim({6, 9}),
                                    platform::CUDAPlace());
    framework::Tensor slice_tensor = src_tensor.Slice(2, 6);
    framework::DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 2);
    EXPECT_EQ(slice_dims[0], 4);
    EXPECT_EQ(slice_dims[1], 9);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.mutable_data<double>(
            src_tensor.dims(), platform::CUDAPlace()));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<double>());
    uintptr_t slice_mutable_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.mutable_data<double>(
            slice_tensor.dims(), platform::CUDAPlace()));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    EXPECT_EQ(slice_data_address, slice_mutable_data_address);
    EXPECT_EQ(src_data_address + 9 * 2 * sizeof(double), slice_data_address);
  }
#endif
}

TEST(Tensor, ReshapeToMatrix) {
  framework::Tensor src;
  int* src_ptr = src.mutable_data<int>({2, 3, 4, 9}, platform::CPUPlace());
  for (int i = 0; i < 2 * 3 * 4 * 9; ++i) {
    src_ptr[i] = i;
  }
  framework::Tensor res = framework::ReshapeToMatrix(src, 2);
  ASSERT_EQ(res.dims()[0], 2 * 3);
  ASSERT_EQ(res.dims()[1], 4 * 9);
}

TEST(Tensor, Layout) {
  framework::Tensor src;
  ASSERT_EQ(src.layout(), framework::DataLayout::kNCHW);
  src.set_layout(framework::DataLayout::kAnyLayout);
  ASSERT_EQ(src.layout(), framework::DataLayout::kAnyLayout);
}
