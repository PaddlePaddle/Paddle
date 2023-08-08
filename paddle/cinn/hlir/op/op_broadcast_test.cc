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

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn {
namespace hlir {
namespace framework {

using CCompute =
    std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

TEST(Operator, Operator_ElementWise_Add_Test0) {
  auto add = Operator::Get("elementwise_add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attrs;
  std::vector<ir::Tensor> inputs{A.tensor(), B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[add](
      attrs, inputs, type, {{M.as_int32(), N.as_int32()}}, target));
  ASSERT_EQ(impl->name, "strategy.elementwise_add.x86");
  ASSERT_EQ(add->description, "elementwise_add function");

  std::string func_name = "add1";
  Module::Builder builder("module0", target);

  std::string out_name = "C";
  common::CINNValuePack cinn_input =
      common::CINNValuePack{{common::CINNValue(A),
                             common::CINNValue(B),
                             common::CINNValue(out_name)}};
  std::vector<std::string> input_output_names{"A", "B", out_name};

  auto funcs = framework::GetFuncFromImpl(
      impl, cinn_input, inputs, input_output_names, func_name, target);

  for (auto func : funcs) {
    LOG(INFO) << "Test Operator_ElementWise_Add_Test0's Strategy, func is :\n"
              << func;
    builder.AddFunction(func);
  }

  auto jit = backends::ExecutionEngine::Create({});
  auto module = builder.Build();
  jit->Link(module);
  auto fn = jit->Lookup("fn_" + func_name);
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);
  cinn_buffer_t *A_buf;
  cinn_buffer_t *B_buf;
  int set_value = 0;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_align(512)
                .set_val(set_value)
                .Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_align(512)
                .set_val(set_value)
                .Build();
  } else {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_align(512)
                .set_random()
                .Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                .set_align(512)
                .set_random()
                .Build();
  }
  auto *C_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                    .set_align(512)
                    .set_zero()
                    .Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf->memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    ASSERT_NEAR(cd[i], ad[i] + bd[i], 1e-5);
  }
}
#ifdef CINN_WITH_CUDA
TEST(Operator, Operator_ElementWise_Add_Test1) {
  auto add = Operator::Get("elementwise_add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {N});

  NodeAttr attrs;
  attrs.attr_store["axis"] = 1;
  std::vector<ir::Tensor> inputs{A.tensor(), B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultNVGPUTarget();
  auto impl = OpStrategy::SelectImpl(
      strategy[add](attrs, inputs, type, {{100, 32}}, target));
  ASSERT_EQ(impl->name, "strategy.elementwise_add.x86");
  ASSERT_EQ(add->description, "elementwise_add function");

  std::string func_name = "add2";
  Module::Builder builder("module", target);

  std::string out_name = "C";
  common::CINNValuePack cinn_input =
      common::CINNValuePack{{common::CINNValue(A),
                             common::CINNValue(B),
                             common::CINNValue(out_name)}};
  std::vector<std::string> input_output_names{"A", "B", out_name};

  auto funcs = framework::GetFuncFromImpl(
      impl, cinn_input, inputs, input_output_names, func_name, target);

  for (auto func : funcs) {
    builder.AddFunction(func);
    LOG(INFO) << "Test Operator_ElementWise_Add_Test1's Strategy, func is :\n"
              << func;
  }

  backends::CodeGenCUDA_Dev codegen(target);

  auto module = builder.Build();
  auto source_code = codegen.Compile(module);
  LOG(INFO) << "Operator_ElementWise_Add_Test1 source code:\n" << source_code;
}
#endif

TEST(Operator, Operator_BroadcastTo) {
  auto broadcast_to = Operator::Get("broadcast_to");
  Operator temp = *broadcast_to;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1);
  Placeholder<float> B("B", {N});

  NodeAttr attrs;
  std::vector<int> out_shape = {16};
  attrs.attr_store["out_shape"] = out_shape;

  std::vector<int> broadcast_axes = {0};
  attrs.attr_store["broadcast_axes"] = broadcast_axes;

  std::vector<ir::Tensor> inputs{B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();

  auto impl = OpStrategy::SelectImpl(
      strategy[broadcast_to](attrs, inputs, type, {out_shape}, target));

  std::string func_name = "broadcast_to";

  std::string out_name = "C";
  common::CINNValuePack cinn_input = common::CINNValuePack{
      {common::CINNValue(B), common::CINNValue(out_name)}};
  std::vector<std::string> input_output_names{"B", out_name};

  auto funcs = framework::GetFuncFromImpl(
      impl, cinn_input, inputs, input_output_names, func_name, target);

  for (auto func : funcs) {
    LOG(INFO) << "Test Operator_BroadcastTo's Strategy, func is :\n" << func;
  }
}

common::CINNValuePack GetComputeResult(
    const std::shared_ptr<OpImpl> &impl,
    std::vector<common::CINNValue> &cinn_inputs,  // NOLINT
    const std::string &output_name = "") {
  cinn_inputs.emplace_back(output_name);
  return impl->fcompute(common::CINNValuePack{cinn_inputs});
}

TEST(Operator, Operator_BroadcastTo_0) {
  auto const_scalar = Operator::Get("const_scalar");
  auto broadcast_to = Operator::Get("broadcast_to");
  auto reduce_sum = Operator::Get("reduce_sum");
  auto elementwise_add = Operator::Get("elementwise_mul");

  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(16);
  Placeholder<float> A("A", {N, N, N, N});

  NodeAttr attrs;
  attrs.attr_store["value"] = 0.5f;

  std::vector<int> out_shape = {16};
  attrs.attr_store["out_shape"] = out_shape;

  std::vector<int> broadcast_axes = {0};
  attrs.attr_store["broadcast_axes"] = broadcast_axes;

  std::vector<int> dim = {0, 2, 3};
  attrs.attr_store["dim"] = dim;

  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();

  auto impl_0 = OpStrategy::SelectImpl(strategy[const_scalar](
      attrs, std::vector<ir::Tensor>{}, type, {out_shape}, target));
  std::vector<common::CINNValue> cinn_inputs;
  common::CINNValuePack rets_0 = GetComputeResult(impl_0, cinn_inputs, "out_0");
  ir::Expr out_0 = rets_0[0];
  auto tensor_0 = out_0.as_tensor_ref();
  poly::StageMap stages_0 = rets_0.back();

  auto impl_1 = OpStrategy::SelectImpl(
      strategy[broadcast_to](attrs, {tensor_0}, type, {out_shape}, target));
  std::vector<common::CINNValue> cinn_inputs_1 = {
      {common::CINNValue(tensor_0)}};
  common::CINNValuePack rets_1 =
      GetComputeResult(impl_1, cinn_inputs_1, "out_1");

  ir::Expr out_1 = rets_1[0];
  auto tensor_1 = out_1.as_tensor_ref();
  poly::StageMap stages_1 = rets_1.back();

  auto impl_2 = OpStrategy::SelectImpl(
      strategy[reduce_sum](attrs, {A.tensor()}, type, {out_shape}, target));
  std::vector<common::CINNValue> cinn_inputs_2 = {
      {common::CINNValue(A.tensor())}};
  common::CINNValuePack rets_2 =
      GetComputeResult(impl_2, cinn_inputs_2, "out_2");

  ir::Expr out_2 = rets_2[0];
  auto tensor_2 = out_2.as_tensor_ref();
  poly::StageMap stages_2 = rets_2.back();

  std::vector<common::CINNValue> cinn_inputs_4 = {
      {common::CINNValue(A.tensor())}};
  common::CINNValuePack rets_4 =
      GetComputeResult(impl_2, cinn_inputs_4, "out_4");
  ir::Expr out_4 = rets_4[0];
  auto tensor_4 = out_4.as_tensor_ref();
  poly::StageMap stages_4 = rets_4.back();

  auto impl_3 = OpStrategy::SelectImpl(strategy[elementwise_add](
      attrs, {tensor_1, tensor_2}, type, {out_shape}, target));
  std::vector<common::CINNValue> cinn_inputs_3 = {
      {common::CINNValue(tensor_1), common::CINNValue(tensor_2)}};
  common::CINNValuePack rets_3 =
      GetComputeResult(impl_3, cinn_inputs_3, "out_3");

  ir::Expr out_3 = rets_3[0];
  auto tensor_3 = out_3.as_tensor_ref();
  poly::StageMap stages_3 = rets_3.back();

  stages_3->InsertLazily(tensor_0, stages_0[tensor_0]);
  stages_3->InsertLazily(tensor_1, stages_1[tensor_1]);
  stages_3->InsertLazily(tensor_2, stages_2[tensor_2]);
  stages_3->InsertLazily(tensor_4, stages_4[tensor_4]);
  stages_3[tensor_0]->ComputeInline();
  stages_3[tensor_1]->ComputeInline();
  stages_3[tensor_2]->SetBuffer("local");
  stages_3[tensor_4]->SimpleComputeAt(stages_3[tensor_2], 3);
  stages_3[tensor_2]->SimpleComputeAt(stages_3[tensor_3], 0);

  std::vector<ir::Tensor> inputs = {A.tensor(), tensor_3, tensor_4};
  auto func = Lower("broadcast_to", stages_3, inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
