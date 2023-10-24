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
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/platform/init_phi.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);

namespace paddle {
namespace framework {

TEST(VJP, TanhBackwardTest) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::TanhOp op2 =
      builder->Build<paddle::dialect::TanhOp>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op2.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op3.out()}};

  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd_op.tanh");
  auto tanh_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  tanh_vjp_interface_impl->vjp_(
      op2.operation(), inputs, outputs, out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_1", prefix_str + "_inner_var_3"});
  test_core.Run({});
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
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::Tanh_Op op2 =
      builder->Build<paddle::dialect::Tanh_Op>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op2.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op3.out()}};

  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd_op.tanh_");
  auto tanh_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  tanh_vjp_interface_impl->vjp_(
      op2.operation(), inputs, outputs, out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_0", prefix_str + "_inner_var_2"});
  test_core.Run({});
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
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::MeanOp op2 =
      builder->Build<paddle::dialect::MeanOp>(op1.out());

  paddle::dialect::FullOp op3 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op2.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op3.out()}};

  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd_op.mean");
  auto mean_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  mean_vjp_interface_impl->vjp_(
      op2.operation(), inputs, outputs, out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars(
      {prefix_str + "_inner_var_1", prefix_str + "_inner_var_3"});
  test_core.Run({});
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

TEST(VJP, ConcatBackwardTest) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 2}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  std::vector<pir::Value> combine_input{{op1.out(), op1.out()}};
  pir::CombineOp op2 = builder->Build<pir::CombineOp>(combine_input);
  paddle::dialect::ConcatOp op3 =
      builder->Build<paddle::dialect::ConcatOp>(op2.out(), 0);

  paddle::dialect::FullOp op4 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  std::vector<std::vector<bool>> stop_gradients{{false, false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out(), op1.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op3.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op4.out()}};
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd_op.concat");
  auto concat_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  concat_vjp_interface_impl->vjp_(
      op3.operation(), inputs, outputs, out_grads, stop_gradients);
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars({prefix_str + "_inner_var_3",
                           prefix_str + "_inner_var_7",
                           prefix_str + "_inner_var_8"});
  test_core.Run({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_3")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_3")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor_0 =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_7")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_7")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor_1 =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_8")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_8")
                ->Get<phi::DenseTensor>();
  ASSERT_EQ(out_tensor.data<float>()[0], 2.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[0], 1.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[1], 1.0);
  ASSERT_EQ(grad_out_tensor_1.data<float>()[0], 1.0);
  ASSERT_EQ(grad_out_tensor_1.data<float>()[1], 1.0);
}

TEST(VJP, AddBackwardTest) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::FullOp op2 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::AddOp op3 =
      builder->Build<paddle::dialect::AddOp>(op1.out(), op2.out());

  paddle::dialect::FullOp op4 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}, {false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out(), op2.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op3.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op4.out()}};

  pir::OpInfo op3_info = ctx->GetRegisteredOpInfo("pd_op.add");
  auto add_vjp_interface_impl =
      op3_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  add_vjp_interface_impl->vjp_(
      op3.operation(), inputs, outputs, out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars({prefix_str + "_inner_var_2",
                           prefix_str + "_inner_var_4",
                           prefix_str + "_inner_var_5"});
  test_core.Run({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_2")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_2")
                ->Get<phi::DenseTensor>();
  auto dx =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_4")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_4")
                ->Get<phi::DenseTensor>();

  auto dy =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_5")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_5")
                ->Get<phi::DenseTensor>();
  ASSERT_EQ(out_tensor.data<float>()[0], 4.0);
  ASSERT_EQ(dx.data<float>()[0], 1.0);
  ASSERT_EQ(dy.data<float>()[0], 1.0);
}

TEST(VJP, Add_BackwardTest) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::FullOp op2 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::Add_Op op3 =
      builder->Build<paddle::dialect::Add_Op>(op1.out(), op2.out());

  paddle::dialect::FullOp op4 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}, {false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out(), op2.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op3.out()}};
  std::vector<std::vector<pir::Value>> out_grads{{op4.out()}};

  pir::OpInfo op3_info = ctx->GetRegisteredOpInfo("pd_op.add_");
  auto add_inplace_vjp_interface_impl =
      op3_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();
  add_inplace_vjp_interface_impl->vjp_(
      op3.operation(), inputs, outputs, out_grads, stop_gradients);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars({prefix_str + "_inner_var_0",
                           prefix_str + "_inner_var_3",
                           prefix_str + "_inner_var_4"});
  test_core.Run({});
  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_0")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_0")
                ->Get<phi::DenseTensor>();
  auto dx =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_3")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_3")
                ->Get<phi::DenseTensor>();

  auto dy =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_4")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_4")
                ->Get<phi::DenseTensor>();
  ASSERT_EQ(out_tensor.data<float>()[0], 4.0);
  ASSERT_EQ(dx.data<float>()[0], 1.0);
  ASSERT_EQ(dy.data<float>()[0], 1.0);
}

TEST(VJP, SplitBackwardTest) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));
  paddle::dialect::APIBuilder::Instance().SetProgram(&program);

  std::shared_ptr<pir::Builder> builder =
      paddle::dialect::APIBuilder::Instance().GetBuilder();
  paddle::dialect::FullOp op1 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 2.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::SplitOp op2 = builder->Build<paddle::dialect::SplitOp>(
      op1.out(), std::vector<int64_t>{1, 1}, 0);

  pir::SplitOp op3 = builder->Build<pir::SplitOp>(op2.out());

  paddle::dialect::FullOp op4 = builder->Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  std::vector<std::vector<bool>> stop_gradients{{false}};
  std::vector<std::vector<pir::Value>> inputs{{op1.out()}};
  std::vector<std::vector<pir::OpResult>> outputs{{op3.outputs()}};
  std::vector<std::vector<pir::Value>> out_grads{{op3.result(0), op4.out()}};
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo("pd_op.split");

  auto concat_vjp_interface_impl =
      op2_info.GetInterfaceImpl<paddle::dialect::VjpInterface>();

  concat_vjp_interface_impl->vjp_(
      op2.operation(), inputs, outputs, out_grads, stop_gradients);
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;
  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);
  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string prefix_str = os.str();
  test_core.SetSkipGcVars({prefix_str + "_inner_var_4",
                           prefix_str + "_inner_var_5",
                           prefix_str + "_inner_var_8"});
  test_core.Run({});
  auto out_tensor_0 =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_4")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_4")
                ->Get<phi::DenseTensor>();
  auto out_tensor_1 =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_5")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_5")
                ->Get<phi::DenseTensor>();
  auto grad_out_tensor_0 =
      test_core.local_scope() == nullptr
          ? scope.FindVar(prefix_str + "_inner_var_8")->Get<phi::DenseTensor>()
          : test_core.local_scope()
                ->FindVar(prefix_str + "_inner_var_8")
                ->Get<phi::DenseTensor>();
  ASSERT_EQ(out_tensor_0.data<float>()[0], 2.0);
  ASSERT_EQ(out_tensor_0.data<float>()[1], 2.0);
  ASSERT_EQ(out_tensor_1.data<float>()[0], 2.0);
  ASSERT_EQ(out_tensor_1.data<float>()[1], 2.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[0], 2.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[1], 2.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[2], 1.0);
  ASSERT_EQ(grad_out_tensor_0.data<float>()[3], 1.0);
}

}  // namespace framework
}  // namespace paddle
