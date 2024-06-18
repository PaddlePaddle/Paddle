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

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace framework {

TEST(test_no_need_buffer_vars_inference, test_static_graph) {
  AttributeMap attrs{{"is_test", true}};
  VariableNameMap inputs;
  VariableNameMap outputs{{"Out", {kEmptyVarName, "tmp_0"}}};

  StaticGraphInferNoNeedBufferVarsContext ctx(inputs, outputs, attrs);

  ASSERT_TRUE(ctx.HasOutput("Out"));
  ASSERT_FALSE(ctx.HasOutput("X"));

  ASSERT_TRUE(PADDLE_GET_CONST(bool, ctx.GetAttr("is_test")));
}

TEST(test_no_need_buffer_vars_inference, test_dygraph) {
  AttributeMap attrs{{"is_test", true}};
  imperative::NameVarMap<imperative::VariableWrapper> inputs;
  imperative::NameVarMap<imperative::VariableWrapper> outputs;
  outputs["Out"].emplace_back(nullptr);
  outputs["Out"].emplace_back(new imperative::VariableWrapper("tmp_0"));

  DyGraphInferNoNeedBufferVarsContext ctx(inputs, outputs, attrs);

  ASSERT_TRUE(ctx.HasOutput("Out"));
  ASSERT_FALSE(ctx.HasOutput("X"));

  ASSERT_TRUE(PADDLE_GET_CONST(bool, ctx.GetAttr("is_test")));
}

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TestNoNeedBufferVarsInferer, "X1", "X2");

TEST(test_no_need_buffer_vars_inference, test_nullptr_comparison) {
  InferNoNeedBufferVarsFN infer_fn;
  ASSERT_FALSE(static_cast<bool>(infer_fn));
  ASSERT_TRUE(!infer_fn);
  ASSERT_TRUE(infer_fn == nullptr);
  ASSERT_TRUE(nullptr == infer_fn);
  ASSERT_FALSE(infer_fn != nullptr);
  ASSERT_FALSE(nullptr != infer_fn);

  infer_fn.Reset(std::make_shared<TestNoNeedBufferVarsInferer>());
  ASSERT_TRUE(static_cast<bool>(infer_fn));
  ASSERT_FALSE(!infer_fn);
  ASSERT_FALSE(infer_fn == nullptr);
  ASSERT_FALSE(nullptr == infer_fn);
  ASSERT_TRUE(infer_fn != nullptr);
  ASSERT_TRUE(nullptr != infer_fn);

  auto no_need_slots =
      infer_fn(VariableNameMap{}, VariableNameMap{}, AttributeMap{});
  ASSERT_EQ(no_need_slots.size(), 2UL);
  ASSERT_EQ(no_need_slots.count("X1"), 1UL);
  ASSERT_EQ(no_need_slots.count("X2"), 1UL);
}

}  // namespace framework
}  // namespace paddle
