// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/core/compat/type_defs.h"

namespace paddle {
namespace imperative {
extern std::string LayerDebugString(const std::string& op_type,
                                    const NameVarMap<egr::EagerTensor>& ins,
                                    const NameVarMap<egr::EagerTensor>& outs);

extern std::shared_ptr<GradOpNode> CreateGradOpNode(
    const framework::OperatorBase& op, const NameTensorMap& ins,
    const NameTensorMap& outs, const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs, const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map);

TEST(test_eager, eager_debug) {
  std::shared_ptr<egr::EagerTensor> x_in(new egr::EagerTensor("x_in"));
  std::shared_ptr<egr::EagerTensor> y_in(new egr::EagerTensor("y_in"));
  std::shared_ptr<egr::EagerTensor> vout(new egr::EagerTensor("vout"));
  imperative::NameVarMap<egr::EagerTensor> ins = {{"X", {x_in}}, {"Y", {y_in}}};
  imperative::NameVarMap<egr::EagerTensor> outs = {{"Out", {vout}}};
  LayerDebugString("mul", ins, outs);
}
TEST(test_create_node, eager_node) {
  auto op = framework::OpRegistry::CreateOp("mul", {}, {}, {}, false);
  framework::Scope scope;
  auto ctx = framework::RuntimeContext({}, {});
  imperative::NameVarMap<egr::EagerTensor> ins = {{"X", {nullptr}},
                                                  {"Y", {nullptr}}};
  imperative::NameVarMap<egr::EagerTensor> outs = {{"Out", {nullptr}}};
  CreateGradOpNode((*op.get()), ins, outs, framework::AttributeMap{},
                   framework::AttributeMap{}, platform::CPUPlace(), {});
}
TEST(test_var_helper, eager_var_helper) {
  framework::Variable var0, var1, var2, var3, var4, var5, var6, var7, var8;
  InitializeVariable(&var0, paddle::framework::proto::VarType::FEED_MINIBATCH);
  InitializeVariable(&var1, paddle::framework::proto::VarType::STEP_SCOPES);
  InitializeVariable(&var2, paddle::framework::proto::VarType::LOD_RANK_TABLE);
  InitializeVariable(&var3,
                     paddle::framework::proto::VarType::LOD_TENSOR_ARRAY);
  InitializeVariable(&var4, paddle::framework::proto::VarType::STRINGS);
  InitializeVariable(&var5, paddle::framework::proto::VarType::VOCAB);
  InitializeVariable(&var6, paddle::framework::proto::VarType::READER);
  InitializeVariable(&var7, paddle::framework::proto::VarType::RAW);
  ASSERT_ANY_THROW(
      InitializeVariable(&var8, paddle::framework::proto::VarType::FP64));

  auto egr_tensor = std::make_shared<egr::EagerTensor>();
  auto egr_tensor2 = std::make_shared<egr::EagerTensor>();
  egr_tensor->MutableVar()
      ->GetMutable<pten::SelectedRows>()
      ->mutable_value()
      ->mutable_data<float>(platform::CPUPlace());
  egr_tensor2->MutableVar()->GetMutable<framework::LoDRankTable>();
  VLOG(6) << "egr_tensor create with ";
  ASSERT_TRUE(platform::is_cpu_place(GetPlace<egr::EagerTensor>(egr_tensor)));
  ASSERT_TRUE(GetDataType<egr::EagerTensor>(egr_tensor) ==
              framework::proto::VarType::FP32);
  GetCachedValue<egr::EagerTensor>(
      egr_tensor, framework::OpKernelType(framework::proto::VarType::FP32,
                                          platform::CPUPlace()));
  SetCachedValue<egr::EagerTensor>(
      egr_tensor, framework::OpKernelType(framework::proto::VarType::FP32,
                                          platform::CPUPlace()),
      egr_tensor2);
  ASSERT_ANY_THROW(GetPlace<egr::EagerTensor>(egr_tensor2));
  ASSERT_ANY_THROW(SetType<egr::EagerTensor>(
      egr_tensor, paddle::framework::proto::VarType::LOD_TENSOR_ARRAY));
}
}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
