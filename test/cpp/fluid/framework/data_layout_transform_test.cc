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
#include "paddle/phi/common/bfloat16.h"

TEST(DataTransform, DataLayoutFunction) {
  auto place = phi::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  phi::DenseTensor out = phi::DenseTensor();
  in.mutable_data<double>(common::make_ddim({2, 3, 1, 2}), place);
  in.set_layout(phi::DataLayout::kNHWC);

  auto kernel_nhwc =
      phi::KernelKey(place, phi::DataLayout::kNHWC, phi::DataType::FLOAT32);
  auto kernel_nchw =
      phi::KernelKey(place, phi::DataLayout::kNCHW, phi::DataType::FLOAT32);

  paddle::framework::TransDataLayout(kernel_nhwc, kernel_nchw, in, &out, place);

  EXPECT_TRUE(out.layout() == phi::DataLayout::kNCHW);
  EXPECT_TRUE(out.dims() == common::make_ddim({2, 2, 3, 1}));

  paddle::framework::TransDataLayout(kernel_nchw, kernel_nhwc, in, &out, place);

  EXPECT_TRUE(in.layout() == phi::DataLayout::kNHWC);
  EXPECT_TRUE(in.dims() == common::make_ddim({2, 3, 1, 2}));
}

#ifdef PADDLE_WITH_DNNL
TEST(DataTransformBf16, GetDataFromTensorDNNL) {
  auto place = phi::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  in.mutable_data<phi::dtype::bfloat16>(common::make_ddim({2, 3, 1, 2}), place);

  void* in_data =
      phi::funcs::GetDataFromTensor(in, dnnl::memory::data_type::bf16);
  EXPECT_EQ(in_data, phi::funcs::to_void_cast(in.data<phi::dtype::bfloat16>()));
}

TEST(DataTransformInt32, GetDataFromTensorDNNL) {
  auto place = phi::CPUPlace();
  phi::DenseTensor in = phi::DenseTensor();
  in.mutable_data<int32_t>(common::make_ddim({2, 3, 1, 2}), place);

  void* in_data =
      phi::funcs::GetDataFromTensor(in, dnnl::memory::data_type::s32);
  EXPECT_EQ(in_data, phi::funcs::to_void_cast(in.data<int32_t>()));
}
#endif
