// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/new_ir_interpreter.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/platform/init_phi.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean_grad, CPU, ALL_LAYOUT);

namespace paddle {
namespace framework {

TEST(VJP, TanhBackwardTest) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<ir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::TanhOp op2 =
      builder->Build<paddle::dialect::TanhOp>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<int>> stop_gradients{{0}};
  std::vector<std::vector<ir::OpResult>> out_grads{{op3.out()}};

  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd.tanh");
  auto tanh_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  tanh_vjp_interface_impl->vjp_(op2.operation(), out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, std::move(kernel_program), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_1", prefix_str + "_inner_var_3"});
  test_core.BetaRun({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_1")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_1")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_3")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_3")
                ->Get<phi::DenseTensor>();

  ASSERT_NEAR(out_tensor.data<float>()[0], 0.76159, 1e-5);
  ASSERT_NEAR(grad_out_tensor.data<float>()[0], 0.83995, 1e-5);
}

TEST(VJP, Tanh_BackwardTest) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<ir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::Tanh_Op op2 =
      builder->Build<paddle::dialect::Tanh_Op>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<int>> stop_gradients{{0}};
  std::vector<std::vector<ir::OpResult>> out_grads{{op3.out()}};

  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd.tanh_");
  auto tanh_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  tanh_vjp_interface_impl->vjp_(op2.operation(), out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, std::move(kernel_program), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_0", prefix_str + "_inner_var_2"});
  test_core.BetaRun({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_0")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_0")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_2")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_2")
                ->Get<phi::DenseTensor>();

  ASSERT_NEAR(out_tensor.data<float>()[0], 0.76159, 1e-5);
  ASSERT_NEAR(grad_out_tensor.data<float>()[0], 0.83995, 1e-5);
}

TEST(VJP, MeanBackwardTest) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<ir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::MeanOp op2 =
      builder->Build<paddle::dialect::MeanOp>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<int>> stop_gradients{{0}};
  std::vector<std::vector<ir::OpResult>> out_grads{{op3.out()}};

  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd.mean");
  auto mean_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  mean_vjp_interface_impl->vjp_(op2.operation(), out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, std::move(kernel_program), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_1", prefix_str + "_inner_var_3"});
  test_core.BetaRun({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_1")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_1")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_3")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_3")
                ->Get<phi::DenseTensor>();
  ASSERT_EQ(out_tensor.data<float>()[0], 2.0);
  ASSERT_EQ(grad_out_tensor.data<float>()[0], 0.25);
  ASSERT_EQ(grad_out_tensor.data<float>()[1], 0.25);
  ASSERT_EQ(grad_out_tensor.data<float>()[2], 0.25);
  ASSERT_EQ(grad_out_tensor.data<float>()[3], 0.25);
}

}  // namespace framework
}  // namespace paddle
