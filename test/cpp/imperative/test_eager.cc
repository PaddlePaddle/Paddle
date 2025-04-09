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
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace imperative {
extern std::string LayerDebugString(const std::string& op_type,
                                    const NameVarMap<egr::EagerVariable>& ins,
                                    const NameVarMap<egr::EagerVariable>& outs);

extern std::shared_ptr<GradOpNode> CreateGradOpNode(
    const framework::OperatorBase& op,
    const NameTensorMap& ins,
    const NameTensorMap& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const phi::Place& place,
    const std::map<std::string, std::string>& inplace_map);

TEST(test_eager, eager_debug) {
  std::shared_ptr<egr::EagerVariable> x_in(new egr::EagerVariable("x_in"));
  std::shared_ptr<egr::EagerVariable> y_in(new egr::EagerVariable("y_in"));
  std::shared_ptr<egr::EagerVariable> vout(new egr::EagerVariable("vout"));
  imperative::NameVarMap<egr::EagerVariable> ins = {{"X", {x_in}},
                                                    {"Y", {y_in}}};
  imperative::NameVarMap<egr::EagerVariable> outs = {{"Out", {vout}}};
  LayerDebugString("mul", ins, outs);
}
TEST(test_create_node, eager_node) {
  auto op = framework::OpRegistry::CreateOp("mul", {}, {}, {}, false);
  framework::Scope scope;
  auto ctx = framework::RuntimeContext({}, {});
  imperative::NameVarMap<egr::EagerVariable> ins = {{"X", {nullptr}},
                                                    {"Y", {nullptr}}};
  imperative::NameVarMap<egr::EagerVariable> outs = {{"Out", {nullptr}}};
  CreateGradOpNode((*op.get()),
                   ins,
                   outs,
                   framework::AttributeMap{},
                   framework::AttributeMap{},
                   phi::CPUPlace(),
                   {});
}
TEST(test_var_helper, eager_var_helper) {
  framework::Variable var0, var1, var3, var4, var5, var6, var7, var8;
  InitializeVariable(&var0, paddle::framework::proto::VarType::FEED_MINIBATCH);
  InitializeVariable(&var1, paddle::framework::proto::VarType::STEP_SCOPES);
  InitializeVariable(&var3,
                     paddle::framework::proto::VarType::DENSE_TENSOR_ARRAY);
  InitializeVariable(&var4, paddle::framework::proto::VarType::STRINGS);
  InitializeVariable(&var5, paddle::framework::proto::VarType::VOCAB);
  InitializeVariable(&var6, paddle::framework::proto::VarType::READER);
  InitializeVariable(&var7, paddle::framework::proto::VarType::RAW);
  ASSERT_ANY_THROW(
      InitializeVariable(&var8, paddle::framework::proto::VarType::FP64));

  auto egr_tensor = std::make_shared<egr::EagerVariable>();
  egr_tensor->MutableVar()
      ->GetMutable<phi::SelectedRows>()
      ->mutable_value()
      ->mutable_data<float>(phi::CPUPlace());
  VLOG(6) << "egr_tensor create with ";
  ASSERT_TRUE(phi::is_cpu_place(GetPlace<egr::EagerVariable>(egr_tensor)));
  ASSERT_TRUE(GetDataType<egr::EagerVariable>(egr_tensor) ==
              framework::proto::VarType::FP32);
  GetCachedValue<egr::EagerVariable>(egr_tensor,
                                     phi::KernelKey(phi::Backend::CPU,
                                                    phi::DataLayout::ALL_LAYOUT,
                                                    phi::DataType::FLOAT32));
  ASSERT_ANY_THROW(SetType<egr::EagerVariable>(
      egr_tensor, paddle::framework::proto::VarType::DENSE_TENSOR_ARRAY));
}
}  // namespace imperative
}  // namespace paddle

USE_OP_ITSELF(mul);
