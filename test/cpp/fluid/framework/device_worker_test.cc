// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/device_worker.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
TEST(DenseTensor, PrintDenseTensor) {
  phi::DenseTensor tensor1;
  tensor1.Resize({2});
  tensor1.mutable_data<float>(phi::CPUPlace());
  tensor1.data<float>()[0] = 0.2;
  tensor1.data<float>()[1] = 0.5;
  std::string res = PrintDenseTensor(&tensor1, -1, 2);
  ASSERT_EQ(res, "access violation");
  res = PrintDenseTensor(&tensor1, 0, 2);
  ASSERT_EQ(res, "0.2,0.5");

  phi::DenseTensor tensor2;
  tensor2.Resize({2});
  tensor2.mutable_data<int64_t>(phi::CPUPlace());
  tensor2.data<int64_t>()[0] = 1;
  tensor2.data<int64_t>()[1] = 2;
  res = PrintDenseTensor(&tensor2, -1, 2);
  ASSERT_EQ(res, "access violation");
  res = PrintDenseTensor(&tensor2, 0, 2);
  ASSERT_EQ(res, "1,2");

  phi::DenseTensor tensor3;
  tensor3.Resize({2});
  tensor3.mutable_data<double>(phi::CPUPlace());
  tensor3.data<double>()[0] = 0.1;
  tensor3.data<double>()[1] = 0.2;
  res = PrintDenseTensor(&tensor3, 0, 2);
  ASSERT_EQ(res, "0.1,0.2");

  phi::DenseTensor tensor4;
  tensor4.Resize({2});
  tensor4.mutable_data<double>(phi::CPUPlace());
  tensor4.data<double>()[0] = 0.1;
  tensor4.data<double>()[1] = 0.2;
  res = "";
  PrintDenseTensor(&tensor4, 0, 2, res);
  // ASSERT_EQ(res, "0.1,0.2");

  phi::DenseTensor tensor5;
  tensor5.Resize({2});
  tensor5.mutable_data<int64_t>(phi::CPUPlace());
  tensor5.data<int64_t>()[0] = 1;
  tensor5.data<int64_t>()[1] = 2;
  res = "";
  PrintDenseTensor(&tensor5, -1, 2, res);
  ASSERT_EQ(res, "access violation");
  res = "";
  PrintDenseTensor(&tensor5, 0, 2, res);
  ASSERT_EQ(res, "1,2");

  phi::DenseTensor tensor6;
  tensor6.Resize({2});
  tensor6.mutable_data<float>(phi::CPUPlace());
  tensor6.data<float>()[0] = 0.2;
  tensor6.data<float>()[1] = 0.5;
  res = "";
  PrintDenseTensor(&tensor6, -1, 2, res);
  // ASSERT_EQ(res, "access violation");
  res = "";
  PrintDenseTensor(&tensor6, 0, 2, res);
  // ASSERT_EQ(res, "0.2,0.5");
}

TEST(DenseTensor, GetTensorBound) {
  LegacyLoD lod{{0, 2}};
  phi::DenseTensor tensor;
  tensor.set_lod(lod);
  tensor.Resize({2, 1});
  tensor.mutable_data<float>(phi::CPUPlace());
  tensor.data<float>()[0] = 0;
  tensor.data<float>()[1] = 1;
  std::pair<int64_t, int64_t> res = GetTensorBound(&tensor, 0);
  ASSERT_EQ(res.first, 0);
  ASSERT_EQ(res.second, 2);
}

TEST(DenseTensor, CheckValidOutput) {
  LegacyLoD lod{{0, 1, 2}};
  phi::DenseTensor tensor;
  tensor.set_lod(lod);
  tensor.Resize({2, 1});
  tensor.mutable_data<float>(phi::CPUPlace());
  tensor.data<float>()[0] = 0;
  tensor.data<float>()[1] = 1;
  ASSERT_TRUE(CheckValidOutput(&tensor, 2));
}

}  // namespace framework
}  // namespace paddle
