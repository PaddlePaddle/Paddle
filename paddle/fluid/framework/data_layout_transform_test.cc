//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/data_layout_transform.h"

#include "gtest/gtest.h"
<<<<<<< HEAD
#include "paddle/fluid/platform/bfloat16.h"

TEST(DataTransform, DataLayoutFunction) {
  auto place = paddle::platform::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  phi::DenseTensor out = phi::DenseTensor();
  in.mutable_data<double>(phi::make_ddim({2, 3, 1, 2}), place);
  in.set_layout(phi::DataLayout::kNHWC);

  auto kernel_nhwc =
      phi::KernelKey(place, phi::DataLayout::kNHWC, phi::DataType::FLOAT32);
  auto kernel_ncwh =
      phi::KernelKey(place, phi::DataLayout::kNCHW, phi::DataType::FLOAT32);

  paddle::framework::TransDataLayout(kernel_nhwc, kernel_ncwh, in, &out, place);

  EXPECT_TRUE(out.layout() == phi::DataLayout::kNCHW);
  EXPECT_TRUE(out.dims() == phi::make_ddim({2, 2, 3, 1}));

  paddle::framework::TransDataLayout(kernel_ncwh, kernel_nhwc, in, &out, place);

  EXPECT_TRUE(in.layout() == phi::DataLayout::kNHWC);
=======

TEST(DataTransform, DataLayoutFunction) {
  auto place = paddle::platform::CPUPlace();
  paddle::framework::Tensor in = paddle::framework::Tensor();
  paddle::framework::Tensor out = paddle::framework::Tensor();
  in.mutable_data<double>(phi::make_ddim({2, 3, 1, 2}), place);
  in.set_layout(paddle::framework::DataLayout::kNHWC);

  auto kernel_nhwc =
      paddle::framework::OpKernelType(paddle::framework::proto::VarType::FP32,
                                      place,
                                      paddle::framework::DataLayout::kNHWC,
                                      paddle::framework::LibraryType::kPlain);
  auto kernel_ncwh =
      paddle::framework::OpKernelType(paddle::framework::proto::VarType::FP32,
                                      place,
                                      paddle::framework::DataLayout::kNCHW,
                                      paddle::framework::LibraryType::kPlain);

  paddle::framework::TransDataLayout(kernel_nhwc, kernel_ncwh, in, &out);

  EXPECT_TRUE(out.layout() == paddle::framework::DataLayout::kNCHW);
  EXPECT_TRUE(out.dims() == phi::make_ddim({2, 2, 3, 1}));

  TransDataLayout(kernel_ncwh, kernel_nhwc, in, &out);

  EXPECT_TRUE(in.layout() == paddle::framework::DataLayout::kNHWC);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  EXPECT_TRUE(in.dims() == phi::make_ddim({2, 3, 1, 2}));
}

#ifdef PADDLE_WITH_MKLDNN
TEST(DataTransformBf16, GetDataFromTensorDNNL) {
  auto place = paddle::platform::CPUPlace();
<<<<<<< HEAD
  phi::DenseTensor in = phi::DenseTensor();
=======
  paddle::framework::Tensor in = paddle::framework::Tensor();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  in.mutable_data<paddle::platform::bfloat16>(phi::make_ddim({2, 3, 1, 2}),
                                              place);

  void* in_data =
<<<<<<< HEAD
      phi::funcs::GetDataFromTensor(in, dnnl::memory::data_type::bf16);
  EXPECT_EQ(in_data,
            phi::funcs::to_void_cast(in.data<paddle::platform::bfloat16>()));
=======
      paddle::framework::GetDataFromTensor(in, dnnl::memory::data_type::bf16);
  EXPECT_EQ(
      in_data,
      paddle::platform::to_void_cast(in.data<paddle::platform::bfloat16>()));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

TEST(DataTransformInt32, GetDataFromTensorDNNL) {
  auto place = paddle::platform::CPUPlace();
<<<<<<< HEAD
  phi::DenseTensor in = phi::DenseTensor();
  in.mutable_data<int32_t>(phi::make_ddim({2, 3, 1, 2}), place);

  void* in_data =
      phi::funcs::GetDataFromTensor(in, dnnl::memory::data_type::s32);
  EXPECT_EQ(in_data, phi::funcs::to_void_cast(in.data<int32_t>()));
=======
  paddle::framework::Tensor in = paddle::framework::Tensor();
  in.mutable_data<int32_t>(phi::make_ddim({2, 3, 1, 2}), place);

  void* in_data =
      paddle::framework::GetDataFromTensor(in, dnnl::memory::data_type::s32);
  EXPECT_EQ(in_data, paddle::platform::to_void_cast(in.data<int32_t>()));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}
#endif
