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
void test_squeeze() {
  framework::Scope scope;
  phi::CPUPlace cpu_place;
  // Prepare Op description
  framework::OpDesc desc;
  // We assume it is kNHWC, so that can use this transformation
  phi::OneDNNContext::tls().set_cur_paddle_data_layout(DataLayout::kNHWC);
  desc.SetType("squeeze2");
  desc.SetInput("X", {"squeeze-X"});
  desc.SetOutput("Out", {"squeeze-Out"});
  // DataLayout::kNHWC will make it become {2, 3, 2, 1}
  AddVarToScope<float>("squeeze-X", &scope, {2, 2, 1, 3});
  AddVarToScope<float>("squeeze-Out", &scope, {2, 3, 2});
  // transform will make it become -1
  std::vector<int> axes({-2});

  desc.SetAttr("axes", axes);
  desc.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(desc);

  op->Run(scope, cpu_place);
}

void test_squeeze2() {
  framework::Scope scope;
  phi::CPUPlace cpu_place;
  // Prepare Op description
  framework::OpDesc desc;
  // We assume it is HNWC, so that can use this transformation
  phi::OneDNNContext::tls().set_cur_paddle_data_layout(DataLayout::kNHWC);
  desc.SetType("squeeze2");
  desc.SetInput("X", {"squeeze-X"});
  desc.SetOutput("Out", {"squeeze-Out"});
  // DataLayout::kNHWC will make it become {2, 1, 3, 2}
  AddVarToScope<float>("squeeze-X", &scope, {2, 3, 2, 1});
  AddVarToScope<float>("squeeze-Out", &scope, {2, 3, 2});
  // transform will make it become -3(1)
  std::vector<int> axes({-1});

  desc.SetAttr("axes", axes);
  desc.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(desc);

  op->Run(scope, cpu_place);
}

TEST(SqueezeOpConverter, normal) { test_squeeze(); }
TEST(SqueezeOpConverter_2, normal) { test_squeeze2(); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
