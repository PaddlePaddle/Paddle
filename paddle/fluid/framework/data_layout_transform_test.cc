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

TEST(DataTransform, DataLayoutFunction) {
  auto place = paddle::platform::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  phi::DenseTensor out = phi::DenseTensor();
  in.mutable_data<double>(phi::make_ddim({2, 3, 1, 2}), place);
  in.set_layout(phi::DataLayout::kNHWC);

  auto kernel_nhwc =
      paddle::framework::OpKernelType(paddle::framework::proto::VarType::FP32,
                                      place,
<<<<<<< HEAD
                                      phi::DataLayout::kNHWC,
=======
                                      paddle::framework::DataLayout::kNHWC,
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                                      paddle::framework::LibraryType::kPlain);
  auto kernel_ncwh =
      paddle::framework::OpKernelType(paddle::framework::proto::VarType::FP32,
                                      place,
<<<<<<< HEAD
                                      phi::DataLayout::kNCHW,
=======
                                      paddle::framework::DataLayout::kNCHW,
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                                      paddle::framework::LibraryType::kPlain);

  paddle::framework::TransDataLayout(kernel_nhwc, kernel_ncwh, in, &out);

  EXPECT_TRUE(out.layout() == phi::DataLayout::kNCHW);
  EXPECT_TRUE(out.dims() == phi::make_ddim({2, 2, 3, 1}));

  TransDataLayout(kernel_ncwh, kernel_nhwc, in, &out);

  EXPECT_TRUE(in.layout() == phi::DataLayout::kNHWC);
  EXPECT_TRUE(in.dims() == phi::make_ddim({2, 3, 1, 2}));
}

#ifdef PADDLE_WITH_MKLDNN
TEST(DataTransformBf16, GetDataFromTensorDNNL) {
  auto place = paddle::platform::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  in.mutable_data<paddle::platform::bfloat16>(phi::make_ddim({2, 3, 1, 2}),
                                              place);

  void* in_data =
      paddle::framework::GetDataFromTensor(in, dnnl::memory::data_type::bf16);
<<<<<<< HEAD
  EXPECT_EQ(in_data,
            phi::funcs::to_void_cast(in.data<paddle::platform::bfloat16>()));
=======
  EXPECT_EQ(
      in_data,
      paddle::platform::to_void_cast(in.data<paddle::platform::bfloat16>()));
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
}

TEST(DataTransformInt32, GetDataFromTensorDNNL) {
  auto place = paddle::platform::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  in.mutable_data<int32_t>(phi::make_ddim({2, 3, 1, 2}), place);

  void* in_data =
      paddle::framework::GetDataFromTensor(in, dnnl::memory::data_type::s32);
  EXPECT_EQ(in_data, phi::funcs::to_void_cast(in.data<int32_t>()));
}
#endif
