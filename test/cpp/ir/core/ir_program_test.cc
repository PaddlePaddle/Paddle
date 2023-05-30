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

#include "paddle/fluid/dialect/pd_dialect.h"
#include "paddle/fluid/dialect/pd_interface.h"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/utils.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

class AddOp : public ir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.add"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static void verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes) {
    if (inputs.size() != 2) {
      throw("The size of inputs must be equal to 2.");
    }
    if (outputs.size() != 1) {
      throw("The size of outputs must be equal to 1.");
    }
  }
};

TEST(program_test, program) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  builtin_dialect->RegisterOp<AddOp>();
  ir::Dialect *paddle_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  // (2) Create an empty program object
  ir::Program program;
  //   ir::Program *program = new ir::Program();
  EXPECT_EQ(program.block()->size() == 0, true);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  ir::Type fp32_dtype = ir::Float32Type::get(ctx);
  paddle::dialect::DenseTensorTypeStorage::Dim dims = {2, 2};
  paddle::dialect::DenseTensorTypeStorage::DataLayout data_layout =
      paddle::dialect::DenseTensorTypeStorage::DataLayout::NCHW;
  paddle::dialect::DenseTensorTypeStorage::LoD lod = {{0, 1, 2}};
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
  std::string op1_name = ir::GetParameterOp::name();
  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::unordered_map<std::string, ir::Attribute> op1_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "a")}};
  ir::Operation *op1 =
      ir::Operation::create({}, {dense_tensor_dtype}, op1_attribute, op1_info);

  program.InsertOp(op1);

  EXPECT_EQ(op1->GetResultByIndex(0).type().dialect().id(),
            paddle_dialect->id());
  using Interface = paddle::dialect::ParameterConvertInterface;
  Interface *a_interface = op1->GetResultByIndex(0)
                               .type()
                               .dialect()
                               .GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> a_var =
      a_interface->ParameterToVariable(program.GetParameter("a"));
  const phi::DenseTensor &a_tensor = a_var->Get<phi::DenseTensor>();
  EXPECT_EQ(a_tensor.numel(), 4);
  EXPECT_EQ(a_tensor.dims(), phi::DDim(dims.data(), dims.size()));
  EXPECT_EQ(a_tensor.dtype(), paddle::dialect::TransToPhiDataType(fp32_dtype));
  EXPECT_EQ(a_tensor.layout(),
            paddle::dialect::TransToPhiDataLayout(data_layout));
  EXPECT_EQ(a_tensor.lod(), lod);
  EXPECT_EQ(a_tensor.offset(), offset);
  for (int64_t i = 0; i < a_tensor.numel(); i++) {
    EXPECT_EQ(*(a_tensor.data<float>() + i), data_a[i]);
  }

  // (5) Def b = GetParameterOp("b"), and create DenseTensor for b.
  std::string op2_name =
      builtin_dialect->name() + "." + std::string(ir::GetParameterOp::name());
  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  std::unordered_map<std::string, ir::Attribute> op2_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "b")}};
  ir::Operation *op2 =
      ir::Operation::create({}, {dense_tensor_dtype}, op2_attribute, op2_info);
  program.InsertOp(op2);

  EXPECT_EQ(op2->GetResultByIndex(0).type().dialect().id(),
            paddle_dialect->id());
  Interface *b_interface = op2->GetResultByIndex(0)
                               .type()
                               .dialect()
                               .GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> b_var =
      b_interface->ParameterToVariable(program.GetParameter("b"));
  const phi::DenseTensor &b_tensor = b_var->Get<phi::DenseTensor>();
  EXPECT_EQ(b_tensor.numel(), 4);
  EXPECT_EQ(b_tensor.dims(), phi::DDim(dims.data(), dims.size()));
  EXPECT_EQ(b_tensor.dtype(), paddle::dialect::TransToPhiDataType(fp32_dtype));
  EXPECT_EQ(b_tensor.layout(),
            paddle::dialect::TransToPhiDataLayout(data_layout));
  EXPECT_EQ(b_tensor.lod(), lod);
  EXPECT_EQ(b_tensor.offset(), offset);
  for (int64_t i = 0; i < b_tensor.numel(); i++) {
    EXPECT_EQ(*(b_tensor.data<float>() + i), data_b[i]);
  }

  // (6) Def c = AddOp(a, b), execute this op.
  std::string op3_name =
      builtin_dialect->name() + "." + std::string(AddOp::name());
  ir::OpInfo op3_info = ctx->GetRegisteredOpInfo(op3_name);
  std::unordered_map<std::string, ir::Attribute> op3_attribute;
  ir::Operation *op3 = ir::Operation::create(
      {op1->GetResultByIndex(0), op2->GetResultByIndex(0)},
      {dense_tensor_dtype},
      op3_attribute,
      op3_info);
  program.InsertOp(op3);

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

  // (7) Def AbsOp(b)
  ir::OpInfo abs_info = ctx->GetRegisteredOpInfo("pd.abs");
  std::vector<ir::OpResult> operands = {op1->GetResultByIndex(0)};
  std::unordered_map<std::string, ir::Attribute> abs_op_attribute;
  std::vector<ir::Type> output_types = {dense_tensor_dtype};
  ir::OperationArgument abs_argument(abs_info);
  abs_argument.addOperands(operands.begin(), operands.end());
  abs_argument.addAttributes(abs_op_attribute.begin(), abs_op_attribute.end());
  abs_argument.addTypes(output_types.begin(), output_types.end());
  ir::Operation *abs_op = ir::Operation::create(std::move(abs_argument));
  paddle::dialect::GetOpInfoInterface interface =
      abs_op->dyn_cast<paddle::dialect::GetOpInfoInterface>();
  EXPECT_EQ(std::get<0>(interface.GetOpInfo())[0].name == "x", true);

  // (8) Def SetParameterOp(c, "c")
  std::string op4_name =
      builtin_dialect->name() + "." + std::string(ir::SetParameterOp::name());
  ir::OpInfo op4_info = ctx->GetRegisteredOpInfo(op4_name);
  std::unordered_map<std::string, ir::Attribute> op4_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "c")}};
  ir::Operation *op4 = ir::Operation::create(
      {op3->GetResultByIndex(0)}, {}, op4_attribute, op4_info);
  program.InsertOp(op4);

  EXPECT_EQ(op4->GetOperandByIndex(0).impl()->source().type().dialect().id(),
            paddle_dialect->id());
  Interface *c_interface = op4->GetOperandByIndex(0)
                               .impl()
                               ->source()
                               .type()
                               .dialect()
                               .GetRegisteredInterface<Interface>();
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
}

TEST(program_test, slice_combine_test) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();

  // (2) Create an empty program object
  ir::Program program;
  //   ir::Program *program = new ir::Program();
  EXPECT_EQ(program.block()->size() == 0, true);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  ir::Type fp32_dtype = ir::Float32Type::get(ctx);

  // (4) Def a = GetParameterOp("a")
  std::string op1_name = ir::GetParameterOp::name();
  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::unordered_map<std::string, ir::Attribute> op1_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "a")}};
  ir::Operation *op1 =
      ir::Operation::create({}, {fp32_dtype}, op1_attribute, op1_info);
  program.InsertOp(op1);

  // (5) Def b = GetParameterOp("b")
  std::string op2_name = std::string(ir::GetParameterOp::name());
  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  std::unordered_map<std::string, ir::Attribute> op2_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "b")}};
  ir::Operation *op2 =
      ir::Operation::create({}, {fp32_dtype}, op2_attribute, op2_info);
  program.InsertOp(op2);

  // (6) Def combine_op = CombineOp("a", "b")
  std::string combine_op_name = std::string(ir::CombineOp::name());
  ir::OpInfo combine_op_info = ctx->GetRegisteredOpInfo(combine_op_name);
  ir::Type output_type =
      ir::VectorType::get(ctx, std::vector<ir::Type>({fp32_dtype, fp32_dtype}));
  ir::Operation *combine_op = ir::Operation::create(
      {op1->GetResultByIndex(0), op2->GetResultByIndex(0)},
      {output_type},
      {},
      combine_op_info);
  program.InsertOp(combine_op);

  // (7) Def slice_op = SliceOp(combine_op, 0)
  std::string slice_op_name = std::string(ir::SliceOp::name());
  ir::OpInfo slice_op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  ir::Attribute index_attr = ir::Int32_tAttribute::get(ctx, 0);
  ir::Operation *slice_op =
      ir::Operation::create({combine_op->GetResultByIndex(0)},
                            {fp32_dtype},
                            {{"index", index_attr}},
                            slice_op_info);
  program.InsertOp(slice_op);

  // (8) Traverse Program
  EXPECT_EQ(program.block()->size() == 4, true);
}
