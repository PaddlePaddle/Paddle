/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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
void test_conv2d_transpose_bias() {
  framework::Scope scope;
  phi::CPUPlace cpu_place;
  // Prepare Op description
  framework::OpDesc desc;

  desc.SetType("conv2d_transpose_bias");
  desc.SetInput("Input", {"convtranspose-Input"});
  desc.SetInput("Filter", {"convtranspose-Filter"});
  desc.SetInput("Bias", {"convtranspose-Bias"});
  desc.SetOutput("Output", {"convtranspose-Out"});

  AddVarToScope<float>("convtranspose-Input", &scope, {1, 512, 23, 19});
  AddVarToScope<float>("convtranspose-Filter", &scope, {512, 256, 5, 5});
  AddVarToScope<float>("convtranspose-Bias", &scope, {256});
  AddVarToScope<float>("convtranspose-Out", &scope, {1, 256, 27, 23});

  desc.SetAttr("use_mkldnn", true);
  desc.SetAttr("is_test", true);

  auto op = paddle::framework::OpRegistry::CreateOp(desc);

  op->Run(scope, cpu_place);
}

TEST(Conv2dTransposeBias, normal) { test_conv2d_transpose_bias(); }

}  // namespace inference
}  // namespace paddle
