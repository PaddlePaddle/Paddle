// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/framework/op.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/runtime/flags.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace framework {

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

TEST(Operator, GetAttrs) {
  auto add      = Operator::Get("elementwise_add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attrs;
  std::vector<ir::Tensor> inputs{A, B};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl             = OpStrategy::SelectImpl(strategy[add](attrs, inputs, type, {{100, 32}}, target));

  ASSERT_EQ(impl->name, "strategy.elementwise_add.x86");
  ASSERT_EQ(add->description, "elementwise_add function");

  std::string func_name = "add1";

  if (FLAGS_cinn_ir_schedule) {
    std::string out_name = "C";
    common::CINNValuePack cinn_input =
        common::CINNValuePack{{common::CINNValue(A), common::CINNValue(B), common::CINNValue(out_name)}};
    std::vector<std::string> input_output_names{"A", "B", out_name};

    auto funcs = framework::GetFuncFromImpl(impl, cinn_input, inputs, input_output_names, func_name, target);

    for (auto func : funcs) {
      LOG(INFO) << "Test Operator_ElementWise_Add_Test0's Strategy, func is :\n" << func;
    }
  } else {
    common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A), common::CINNValue(B)}};
    common::CINNValuePack rets       = impl->fcompute(cinn_input);
    ASSERT_EQ(rets.size(), 2UL);
    rets = impl->fschedule(rets);
    ASSERT_EQ(rets.size(), 2UL);
    // the last element is a StageMap
    for (int i = 0; i < rets->size() - 1; i++) {
      ir::Expr temp = rets[i];
      inputs.push_back(temp.as_tensor_ref());
    }
    auto func = Lower(func_name, rets.back(), inputs);
    LOG(INFO) << "Test Strategy Codegen:\n" << func;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
