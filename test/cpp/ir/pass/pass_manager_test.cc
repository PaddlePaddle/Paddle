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
#include "glog/logging.h"

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

#ifndef _WIN32
class TestAnalysis1 {};
class TestAnalysis2 {};

IR_DECLARE_EXPLICIT_TYPE_ID(TestAnalysis1)
IR_DEFINE_EXPLICIT_TYPE_ID(TestAnalysis1)
IR_DECLARE_EXPLICIT_TYPE_ID(TestAnalysis2)
IR_DEFINE_EXPLICIT_TYPE_ID(TestAnalysis2)

TEST(pass_manager, PreservedAnalyses) {
  ir::detail::PreservedAnalyses pa;
  CHECK_EQ(pa.IsNone(), true);

  CHECK_EQ(pa.IsPreserved<TestAnalysis1>(), false);
  pa.Preserve<TestAnalysis1>();
  CHECK_EQ(pa.IsPreserved<TestAnalysis1>(), true);
  pa.Unpreserve<TestAnalysis1>();
  CHECK_EQ(pa.IsPreserved<TestAnalysis1>(), false);
  CHECK_EQ(pa.IsPreserved<TestAnalysis2>(), false);
  pa.Preserve<TestAnalysis1, TestAnalysis2>();
  CHECK_EQ(pa.IsPreserved<TestAnalysis1>(), true);
  CHECK_EQ(pa.IsPreserved<TestAnalysis2>(), true);
  CHECK_EQ(pa.IsAll(), false);
  pa.PreserveAll();
  CHECK_EQ(pa.IsAll(), true);
  CHECK_EQ(pa.IsNone(), false);
}
#endif

class AddOp : public ir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.add"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void Verify();
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::OpResult l_operand,
                    ir::OpResult r_operand,
                    ir::Type sum_type);
};
void AddOp::Verify() {
  if (num_operands() != 2) {
    throw("The size of inputs must be equal to 2.");
  }
  if (num_results() != 1) {
    throw("The size of outputs must be equal to 1.");
  }
}
void AddOp::Build(ir::Builder &,
                  ir::OperationArgument &argument,
                  ir::OpResult l_operand,
                  ir::OpResult r_operand,
                  ir::Type sum_type) {
  argument.AddOperand(l_operand);
  argument.AddOperand(r_operand);
  argument.AddOutput(sum_type);
}
IR_DECLARE_EXPLICIT_TYPE_ID(AddOp)
IR_DEFINE_EXPLICIT_TYPE_ID(AddOp)

struct CountOpAnalysis {
  explicit CountOpAnalysis(ir::Operation *container_op) {
    IR_ENFORCE(container_op->num_regions() > 0,
               "op must be a container with zero or multiple regions.");

    LOG(INFO) << "In CountOpAnalysis, op is " << container_op->name() << "\n";
    for (size_t i = 0; i < container_op->num_regions(); ++i) {
      auto &region = container_op->region(i);
      for (auto it = region.begin(); it != region.end(); ++it) {
        auto *block = *it;
        for (auto it = block->begin(); it != block->end(); ++it) {
          ++count;
        }
      }
    }

    LOG(INFO) << "-- count is " << count << "\n";
  }

  int count = 0;
};

IR_DECLARE_EXPLICIT_TYPE_ID(CountOpAnalysis)
IR_DEFINE_EXPLICIT_TYPE_ID(CountOpAnalysis)

class TestPass : public ir::Pass {
 public:
  TestPass() : ir::Pass("TestPass", 1) {}
  void Run(ir::Operation *op) override {
    auto count_op_analysis = analysis_manager().GetAnalysis<CountOpAnalysis>();
    pass_state().preserved_analyses.Preserve<CountOpAnalysis>();
    CHECK_EQ(pass_state().preserved_analyses.IsPreserved<CountOpAnalysis>(),
             true);
    CHECK_EQ(count_op_analysis.count, 4);

    auto module_op = op->dyn_cast<ir::ModuleOp>();
    CHECK_EQ(module_op.operation(), op);
    CHECK_EQ(module_op.name(), module_op->name());
    LOG(INFO) << "In " << pass_info().name << ": " << module_op->name()
              << std::endl;

    pass_state().preserved_analyses.Unpreserve<CountOpAnalysis>();
    CHECK_EQ(pass_state().preserved_analyses.IsPreserved<CountOpAnalysis>(),
             false);
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

TEST(pass_manager, PassManager) {
  //
  // TODO(liuyuanle): remove test code other than pass manager
  //

  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  builtin_dialect->RegisterOp<AddOp>();
  ir::Dialect *paddle_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  // (2) Create an empty program object
  ir::Program program(ctx);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  ir::Type fp32_dtype = ir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  ir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  std::vector<float> data_a = {1, 2, 3, 4};
  std::unique_ptr<ir::Parameter> parameter_a =
      std::make_unique<ir::Parameter>(reinterpret_cast<void *>(data_a.data()),
                                      4 * sizeof(float),
                                      dense_tensor_dtype);
  program.SetParameter("a", std::move(parameter_a));
  EXPECT_EQ(program.parameters_num() == 1, true);

  std::vector<float> data_b = {5, 6, 7, 8};
  std::unique_ptr<ir::Parameter> parameter_b =
      std::make_unique<ir::Parameter>(reinterpret_cast<void *>(data_b.data()),
                                      4 * sizeof(float),
                                      dense_tensor_dtype);
  program.SetParameter("b", std::move(parameter_b));
  EXPECT_EQ(program.parameters_num() == 2, true);

  // (4) Def a = GetParameterOp("a"), and create DenseTensor for a.
  ir::Builder builder(ctx, program.block());
  auto op1 = builder.Build<ir::GetParameterOp>("a", dense_tensor_dtype);

  EXPECT_EQ(&program, op1->GetParentProgram());
  EXPECT_EQ(op1->result(0).type().dialect().id(), paddle_dialect->id());
  using Interface = paddle::dialect::ParameterConvertInterface;
  Interface *a_interface =
      op1->result(0).type().dialect().GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> a_var =
      a_interface->ParameterToVariable(program.GetParameter("a"));
  const phi::DenseTensor &a_tensor = a_var->Get<phi::DenseTensor>();
  EXPECT_EQ(a_tensor.numel(), 4);
  EXPECT_EQ(a_tensor.dims(), dims);
  EXPECT_EQ(a_tensor.dtype(), paddle::dialect::TransToPhiDataType(fp32_dtype));
  EXPECT_EQ(a_tensor.layout(), data_layout);
  EXPECT_EQ(a_tensor.lod(), lod);
  EXPECT_EQ(a_tensor.offset(), offset);
  for (int64_t i = 0; i < a_tensor.numel(); i++) {
    EXPECT_EQ(*(a_tensor.data<float>() + i), data_a[i]);
  }

  // (5) Def b = GetParameterOp("b"), and create DenseTensor for b.
  auto op2 = builder.Build<ir::GetParameterOp>("b", dense_tensor_dtype);
  EXPECT_EQ(op2->result(0).type().dialect().id(), paddle_dialect->id());
  Interface *b_interface =
      op2->result(0).type().dialect().GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> b_var =
      b_interface->ParameterToVariable(program.GetParameter("b"));
  const phi::DenseTensor &b_tensor = b_var->Get<phi::DenseTensor>();
  EXPECT_EQ(b_tensor.numel(), 4);
  EXPECT_EQ(b_tensor.dims(), dims);
  EXPECT_EQ(b_tensor.dtype(), paddle::dialect::TransToPhiDataType(fp32_dtype));
  EXPECT_EQ(b_tensor.layout(), data_layout);
  EXPECT_EQ(b_tensor.lod(), lod);
  EXPECT_EQ(b_tensor.offset(), offset);
  for (int64_t i = 0; i < b_tensor.numel(); i++) {
    EXPECT_EQ(*(b_tensor.data<float>() + i), data_b[i]);
  }

  // (6) Def c = AddOp(a, b), execute this op.
  auto op3 =
      builder.Build<AddOp>(op1->result(0), op2->result(0), dense_tensor_dtype);
  phi::CPUContext *dev_ctx = static_cast<phi::CPUContext *>(
      paddle::platform::DeviceContextPool::Instance().Get(
          paddle::platform::CPUPlace()));
  phi::DenseTensor c_tensor =
      phi::Add<float, phi::CPUContext>(*dev_ctx, a_tensor, b_tensor);
  std::shared_ptr<paddle::framework::Variable> variable_c =
      std::make_shared<paddle::framework::Variable>();
  auto *dst_tensor = variable_c->GetMutable<phi::DenseTensor>();
  *dst_tensor = c_tensor;
  EXPECT_EQ(dst_tensor->numel(), b_tensor.numel());
  EXPECT_EQ(dst_tensor->dims(), b_tensor.dims());
  EXPECT_EQ(dst_tensor->dtype(), b_tensor.dtype());
  EXPECT_EQ(dst_tensor->layout(), b_tensor.layout());
  EXPECT_EQ(dst_tensor->lod(), b_tensor.lod());
  EXPECT_EQ(dst_tensor->offset(), b_tensor.offset());
  for (int64_t i = 0; i < dst_tensor->numel(); i++) {
    EXPECT_EQ(*(dst_tensor->data<float>() + i), data_a[i] + data_b[i]);
  }

  // (7) Def SetParameterOp(c, "c")
  auto op4 = builder.Build<ir::SetParameterOp>(op3->result(0), "c");
  EXPECT_EQ(op4->operand(0).type().dialect().id(), paddle_dialect->id());
  Interface *c_interface =
      op4->op_operand(0).type().dialect().GetRegisteredInterface<Interface>();
  //   ir::Parameter *parameter_c =
  //       c_interface->VariableToParameter(variable_c.get());

  std::unique_ptr<ir::Parameter> parameter_c =
      c_interface->VariableToParameter(variable_c.get());
  EXPECT_EQ(parameter_c->type(), dense_tensor_dtype);
  for (int64_t i = 0; i < dst_tensor->numel(); i++) {
    EXPECT_EQ(*(dst_tensor->data<float>() + i),
              *(static_cast<float *>(parameter_c->data()) + i));
  }
  program.SetParameter("c", std::move(parameter_c));

  // (8) Traverse Program
  EXPECT_EQ(program.block()->size() == 4, true);
  EXPECT_EQ(program.parameters_num() == 3, true);

  //
  // TODO(liuyuanle): remove the code above.
  //

  // (9) Test pass manager for program.
  ir::PassManager pm(ctx);

  pm.AddPass(std::make_unique<TestPass>());

  pm.EnableIRPrinting(std::make_unique<ir::PassManager::IRPrinterOption>(
      [](ir::Pass *pass, ir::Operation *op) {
        return pass->name() == "TestPass";
      },
      [](ir::Pass *pass, ir::Operation *op) {
        return pass->name() == "TestPass";
      },
      true,
      true));

  pm.EnablePassTiming(true);

  CHECK_EQ(pm.Run(&program), true);
}
