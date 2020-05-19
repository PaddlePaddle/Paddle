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
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {
TEST(LodTensor, PrintLodTensor) {
  LoDTensor tensor1;
  tensor1.Resize({2});
  tensor1.mutable_data<float>(platform::CPUPlace());
  tensor1.data<float>()[0] = 0.2;
  tensor1.data<float>()[1] = 0.5;
  std::string res = PrintLodTensor(&tensor1, -1, 2);
  ASSERT_EQ(res, "access violation");
  res = PrintLodTensor(&tensor1, 0, 2);
  ASSERT_EQ(res, ":0.2:0.5");

  LoDTensor tensor2;
  tensor2.Resize({2});
  tensor2.mutable_data<int64_t>(platform::CPUPlace());
  tensor2.data<int64_t>()[0] = 1;
  tensor2.data<int64_t>()[1] = 2;
  res = PrintLodTensor(&tensor2, -1, 2);
  ASSERT_EQ(res, "access violation");
  res = PrintLodTensor(&tensor2, 0, 2);
  ASSERT_EQ(res, ":1:2");

  LoDTensor tensor3;
  tensor3.Resize({2});
  tensor3.mutable_data<double>(platform::CPUPlace());
  tensor3.data<double>()[0] = 0.1;
  tensor3.data<double>()[1] = 0.2;
  res = PrintLodTensor(&tensor3, 0, 2);
  ASSERT_EQ(res, ":0.1:0.2");
}

TEST(LodTensor, GetTensorBound) {
  LoD lod{{0, 2}};
  LoDTensor tensor;
  tensor.set_lod(lod);
  tensor.Resize({2, 1});
  tensor.mutable_data<float>(platform::CPUPlace());
  tensor.data<float>()[0] = 0;
  tensor.data<float>()[1] = 1;
  std::pair<int64_t, int64_t> res = GetTensorBound(&tensor, 0);
  ASSERT_EQ(res.first, 0);
  ASSERT_EQ(res.second, 2);
}

TEST(LodTensor, CheckValidOutput) {
  LoD lod{{0, 1, 2}};
  LoDTensor tensor;
  tensor.set_lod(lod);
  tensor.Resize({2, 1});
  tensor.mutable_data<float>(platform::CPUPlace());
  tensor.data<float>()[0] = 0;
  tensor.data<float>()[1] = 1;
  ASSERT_TRUE(CheckValidOutput(&tensor, 2));
}

}  // namespace framework
}  // namespace paddle
