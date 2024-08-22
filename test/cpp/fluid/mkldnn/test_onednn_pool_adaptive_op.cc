/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include <gtest/gtest.h>

#include <fstream>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace inference {
namespace tensorrt {

template <typename DataType>
void AddVarToScope(const std::string var_name,
                   paddle::framework::Scope* scope,
                   const phi::DDim& dims) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0, 100);

  phi::DenseTensor tmp_tensor;
  auto* tmp_data = tmp_tensor.mutable_data<DataType>(dims, phi::CPUPlace());
  auto* tensor = scope->Var(var_name)->GetMutable<phi::DenseTensor>();
  tensor->mutable_data<DataType>(dims, phi::CPUPlace());
  for (auto i = 0; i < tensor->numel(); ++i) {
    tmp_data[i] = static_cast<DataType>(dist(engine));
  }
  paddle::framework::TensorCopySync(tmp_tensor, phi::CPUPlace(), tensor);
}
void test_pool2d(bool adaptive, bool ceil_mode, std::string pool_type = "max") {
  framework::Scope scope;
  phi::CPUPlace cpu_place;

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d-X"});
  desc.SetOutput("Out", {"pool2d-Out"});
  AddVarToScope<float>("pool2d-X", &scope, {1, 3, 9, 12});
  AddVarToScope<float>("pool2d-Out", &scope, {1, 3, 2, 2});
  std::vector<int> ksize({2, 2});
  std::vector<int> strides({1, 1});
  std::vector<int> paddings({0, 0});
  std::string pooling_t = pool_type;

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("adaptive", adaptive);
  desc.SetAttr("ceil_mode", ceil_mode);
  desc.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(desc);

  op->Run(scope, cpu_place);
}

TEST(Pool2dOpConverter, normal) { test_pool2d(false, false); }
TEST(Pool2dOpConverter, adaptive) { test_pool2d(true, false); }

TEST(Pool2dOpConverter, max_ceil_test) { test_pool2d(false, true); }
TEST(Pool2dOpConverter, avg_ceil_test) { test_pool2d(true, true, "avg"); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
