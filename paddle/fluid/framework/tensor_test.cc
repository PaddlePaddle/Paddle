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

namespace paddle {
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

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
  } catch (platform::EnforceNotMet& err) {
    caught = true;
    std::string ex_msg = err.what();
    EXPECT_TRUE(ex_msg.find("Tensor holds no memory. Call "
                            "Tensor::mutable_data firstly.") !=
                std::string::npos);
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
    auto p1_holder = src_tensor.Holder();
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({3, 4}),
                                        platform::CPUPlace());
    EXPECT_NE(p2, nullptr);
    auto p2_holder1 = src_tensor.Holder();
    EXPECT_NE(p1_holder.get(), p2_holder1.get());
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2, 3}),
                                        platform::CPUPlace());
    auto p2_holder2 = src_tensor.Holder();
    EXPECT_EQ(p2_holder1.get(), p2_holder2.get());
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2}),
                                        platform::CPUPlace());
    auto p2_holder3 = src_tensor.Holder();
    EXPECT_EQ(p1, p2);
    EXPECT_EQ(p2_holder2.get(), p2_holder3.get());

    float* p3 = nullptr;
    float* p4 = nullptr;
    // set src_tensor a different type but smaller size.
    // memory block is supposed to be unchanged.
    auto* tmp = src_tensor.mutable_data<uint8_t>(framework::make_ddim({2, 2}),
                                                 platform::CPUPlace());
    p3 = reinterpret_cast<float*>(tmp);
    auto p3_holder1 = src_tensor.Holder();
    EXPECT_EQ(p1, p3);
    EXPECT_EQ(p2_holder3.get(), p3_holder1.get());

    // set src_tensor a different type but bigger size.
    // memory block is supposed to be changed.
    auto* tmp2 = src_tensor.mutable_data<double>(
        framework::make_ddim({2, 2, 3}), platform::CPUPlace());
    auto p3_holder2 = src_tensor.Holder();
    p4 = reinterpret_cast<float*>(tmp2);
    EXPECT_NE(p1, p4);
    EXPECT_NE(p3_holder1.get(), p3_holder2.get());
  }
  // Not sure if it's desired, but currently, Tensor type can be changed.
  {
    framework::Tensor src_tensor;
    int8_t* p1 = src_tensor.mutable_data<int8_t>(framework::make_ddim({1}),
                                                 platform::CPUPlace());
    EXPECT_NE(p1, nullptr);
    *p1 = 1;

    uint8_t* p2 = src_tensor.mutable_data<uint8_t>(framework::make_ddim({1}),
                                                   platform::CPUPlace());
    EXPECT_NE(p2, nullptr);
    EXPECT_EQ(static_cast<int>(p2[0]), 1);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    framework::Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({1, 2, 3}),
                                        platform::CUDAPlace(0));
    auto p1_holder = src_tensor.Holder();
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({3, 1024}),
                                        platform::CUDAPlace(0));
    auto p2_holder = src_tensor.Holder();
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1_holder.get(), p2_holder.get());
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2, 3}),
                                        platform::CUDAPlace(0));
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2}),
                                        platform::CUDAPlace(0));
    EXPECT_EQ(p1, p2);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  {
    framework::Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({1, 2, 3}),
                                        platform::NPUPlace(0));
    auto p1_holder = src_tensor.Holder();
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({3, 1024}),
                                        platform::NPUPlace(0));
    auto p2_holder = src_tensor.Holder();
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1_holder.get(), p2_holder.get());
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2, 3}),
                                        platform::NPUPlace(0));
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(framework::make_ddim({2, 2}),
                                        platform::NPUPlace(0));
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
    } catch (paddle::platform::EnforceNotMet& err) {
      caught = true;
      std::string ex_msg = err.what();
      EXPECT_TRUE(ex_msg.find("Tensor holds no memory. Call "
                              "Tensor::mutable_data firstly.") !=
                  std::string::npos);
    }
    ASSERT_TRUE(caught);

    src_tensor.mutable_data<int>(framework::make_ddim({2, 3, 4}),
                                 platform::CPUPlace());
    dst_tensor.ShareDataWith(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    framework::Tensor src_tensor;
    framework::Tensor dst_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({2, 3, 4}),
                                 platform::CUDAPlace(0));
    dst_tensor.ShareDataWith(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  {
    framework::Tensor src_tensor;
    framework::Tensor dst_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({2, 3, 4}),
                                 platform::NPUPlace(0));
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<double>(framework::make_ddim({6, 9}),
                                    platform::CUDAPlace(0));
    framework::Tensor slice_tensor = src_tensor.Slice(2, 6);
    framework::DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 2);
    EXPECT_EQ(slice_dims[0], 4);
    EXPECT_EQ(slice_dims[1], 9);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.mutable_data<double>(
            src_tensor.dims(), platform::CUDAPlace(0)));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<double>());
    uintptr_t slice_mutable_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.mutable_data<double>(
            slice_tensor.dims(), platform::CUDAPlace(0)));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    EXPECT_EQ(slice_data_address, slice_mutable_data_address);
    EXPECT_EQ(src_data_address + 9 * 2 * sizeof(double), slice_data_address);
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<double>(framework::make_ddim({6, 9}),
                                    platform::NPUPlace(0));
    framework::Tensor slice_tensor = src_tensor.Slice(2, 6);
    framework::DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 2);
    EXPECT_EQ(slice_dims[0], 4);
    EXPECT_EQ(slice_dims[1], 9);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.mutable_data<double>(
            src_tensor.dims(), platform::NPUPlace(0)));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<double>());
    uintptr_t slice_mutable_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.mutable_data<double>(
            slice_tensor.dims(), platform::NPUPlace(0)));
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

TEST(Tensor, FP16) {
  using platform::float16;
  framework::Tensor src;
  float16* src_ptr = src.mutable_data<float16>({2, 3}, platform::CPUPlace());
  for (int i = 0; i < 2 * 3; ++i) {
    src_ptr[i] = static_cast<float16>(i);
  }
  EXPECT_EQ(src.memory_size(), 2 * 3 * sizeof(float16));
  // EXPECT a human readable error message
  // src.data<uint8_t>();
  // Tensor holds the wrong type, it holds N6paddle8platform7float16E at
  // [/paddle/Paddle/paddle/fluid/framework/tensor_impl.h:43]
}

TEST(Tensor, Split) {
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({6, 2}),
                                 platform::CPUPlace());
    std::vector<framework::Tensor> split_tensor_list = src_tensor.Split(2, 0);
    ASSERT_EQ(split_tensor_list.size(), 3UL);
    EXPECT_EQ(split_tensor_list[0].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[0].dims()[1], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[1], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[1], 2);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<int>());
    uintptr_t src_mutable_data_address = reinterpret_cast<uintptr_t>(
        src_tensor.mutable_data<int>(src_tensor.dims(), platform::CPUPlace()));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    for (int i = 0; i < 3; ++i) {
      uintptr_t split_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].data<int>());
      uintptr_t split_mutable_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].mutable_data<int>(
              split_tensor_list[i].dims(), platform::CPUPlace()));
      EXPECT_EQ(split_data_address, split_mutable_data_address);
      EXPECT_EQ(src_data_address + 2 * 2 * i * sizeof(int), split_data_address);
    }
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<double>(framework::make_ddim({6, 4}),
                                    platform::CUDAPlace(0));
    std::vector<framework::Tensor> split_tensor_list = src_tensor.Split(2, 0);
    ASSERT_EQ(split_tensor_list.size(), 3UL);
    EXPECT_EQ(split_tensor_list[0].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[0].dims()[1], 4);
    EXPECT_EQ(split_tensor_list[1].dims()[1], 4);
    EXPECT_EQ(split_tensor_list[2].dims()[1], 4);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.mutable_data<double>(
            src_tensor.dims(), platform::CUDAPlace(0)));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    for (int i = 0; i < 3; ++i) {
      uintptr_t split_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].data<double>());
      uintptr_t split_mutable_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].mutable_data<double>(
              split_tensor_list[i].dims(), platform::CUDAPlace(0)));
      EXPECT_EQ(split_data_address, split_mutable_data_address);
      EXPECT_EQ(src_data_address + 2 * 4 * i * sizeof(double),
                split_data_address);
    }
  }
#endif
}

TEST(Tensor, Chunk) {
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<int>(framework::make_ddim({6, 2}),
                                 platform::CPUPlace());
    std::vector<framework::Tensor> split_tensor_list = src_tensor.Chunk(3, 0);
    ASSERT_EQ(split_tensor_list.size(), 3UL);
    EXPECT_EQ(split_tensor_list[0].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[0].dims()[1], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[1], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[1], 2);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<int>());
    uintptr_t src_mutable_data_address = reinterpret_cast<uintptr_t>(
        src_tensor.mutable_data<int>(src_tensor.dims(), platform::CPUPlace()));
    for (int i = 0; i < 3; ++i) {
      uintptr_t split_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].data<int>());
      uintptr_t split_mutable_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].mutable_data<int>(
              split_tensor_list[i].dims(), platform::CPUPlace()));
      EXPECT_EQ(src_data_address, src_mutable_data_address);
      EXPECT_EQ(split_data_address, split_mutable_data_address);
      EXPECT_EQ(src_data_address + 2 * 2 * i * sizeof(int), split_data_address);
    }
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    framework::Tensor src_tensor;
    src_tensor.mutable_data<double>(framework::make_ddim({6, 4}),
                                    platform::CUDAPlace(0));
    std::vector<framework::Tensor> split_tensor_list = src_tensor.Chunk(3, 0);
    ASSERT_EQ(split_tensor_list.size(), 3UL);
    EXPECT_EQ(split_tensor_list[0].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[1].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[2].dims()[0], 2);
    EXPECT_EQ(split_tensor_list[0].dims()[1], 4);
    EXPECT_EQ(split_tensor_list[1].dims()[1], 4);
    EXPECT_EQ(split_tensor_list[2].dims()[1], 4);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.mutable_data<double>(
            src_tensor.dims(), platform::CUDAPlace(0)));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    for (int i = 0; i < 3; ++i) {
      uintptr_t split_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].data<double>());
      uintptr_t split_mutable_data_address =
          reinterpret_cast<uintptr_t>(split_tensor_list[i].mutable_data<double>(
              split_tensor_list[i].dims(), platform::CUDAPlace(0)));
      EXPECT_EQ(split_data_address, split_mutable_data_address);
      EXPECT_EQ(src_data_address + 2 * 4 * i * sizeof(double),
                split_data_address);
    }
  }
#endif
}
