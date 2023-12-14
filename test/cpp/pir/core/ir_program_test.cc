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

#include <sstream>

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"
// NOTE(zhangbo9674): File pd_op.h is generated by op_gen.py, see details in
// paddle/fluid/pir/dialect/CMakeLists.txt.
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/transforms/param_to_variable.h"

class AddOp : public pir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.add"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void VerifySig();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type sum_type);
};
void AddOp::VerifySig() {
  if (num_operands() != 2) {
    throw("The size of inputs must be equal to 2.");
  }
  if (num_results() != 1) {
    throw("The size of outputs must be equal to 1.");
  }
}
void AddOp::Build(pir::Builder &,
                  pir::OperationArgument &argument,
                  pir::Value l_operand,
                  pir::Value r_operand,
                  pir::Type sum_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(sum_type);
}
IR_DECLARE_EXPLICIT_TYPE_ID(AddOp)
IR_DEFINE_EXPLICIT_TYPE_ID(AddOp)

TEST(program_test, program) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  builtin_dialect->RegisterOp<AddOp>();
  pir::Dialect *paddle_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  // (2) Create an empty program object
  pir::Program program(ctx);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  std::vector<float> data_a = {1, 2, 3, 4};
  std::unique_ptr<pir::Parameter> parameter_a =
      std::make_unique<pir::Parameter>(reinterpret_cast<void *>(data_a.data()),
                                       4 * sizeof(float),
                                       dense_tensor_dtype);
  program.SetParameter("a", std::move(parameter_a));
  EXPECT_EQ(program.parameters_num() == 1, true);

  std::vector<float> data_b = {5, 6, 7, 8};
  std::unique_ptr<pir::Parameter> parameter_b =
      std::make_unique<pir::Parameter>(reinterpret_cast<void *>(data_b.data()),
                                       4 * sizeof(float),
                                       dense_tensor_dtype);
  program.SetParameter("b", std::move(parameter_b));
  EXPECT_EQ(program.parameters_num() == 2, true);

  // (4) Def a = ParameterOp("a"), and create DenseTensor for a.
  pir::Builder builder(ctx, program.block());
  auto op1 = builder.Build<pir::ParameterOp>("a", dense_tensor_dtype);

  EXPECT_EQ(&program, op1->GetParentProgram());
  EXPECT_EQ(op1->result_type(0).dialect().id(), paddle_dialect->id());
  using Interface = paddle::dialect::ParameterConvertInterface;
  Interface *a_interface =
      op1->result_type(0).dialect().GetRegisteredInterface<Interface>();
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

  // (5) Def b = ParameterOp("b"), and create DenseTensor for b.
  auto op2 = builder.Build<pir::ParameterOp>("b", dense_tensor_dtype);

  EXPECT_EQ(op2->result_type(0).dialect().id(), paddle_dialect->id());
  Interface *b_interface =
      op2->result_type(0).dialect().GetRegisteredInterface<Interface>();
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

  // (7) Def AbsOp(b)
  auto abs_op = builder.Build<paddle::dialect::AbsOp>(op1->result(0));
  paddle::dialect::OpYamlInfoInterface interface =
      abs_op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
  EXPECT_EQ(std::get<0>(interface.GetOpInfo())[0].name == "x", true);

  // (8) Def SetParameterOp(c, "c")
  auto op4 = builder.Build<pir::SetParameterOp>(op3->result(0), "c");

  EXPECT_EQ(op4->operand(0).type().dialect().id(), paddle_dialect->id());
  Interface *c_interface =
      op4->operand(0).type().dialect().GetRegisteredInterface<Interface>();
  //   pir::Parameter *parameter_c =
  //       c_interface->VariableToParameter(variable_c.get());
  std::unique_ptr<pir::Parameter> parameter_c =
      c_interface->VariableToParameter(variable_c.get());
  EXPECT_EQ(parameter_c->type(), dense_tensor_dtype);
  for (int64_t i = 0; i < dst_tensor->numel(); i++) {
    EXPECT_EQ(*(dst_tensor->data<float>() + i),
              *(static_cast<float *>(parameter_c->data()) + i));
  }
  program.SetParameter("c", std::move(parameter_c));

  // (8) Traverse Program
  EXPECT_EQ(program.block()->size() == 5, true);
  EXPECT_EQ(program.parameters_num() == 3, true);

  std::stringstream ss;
  program.Print(ss);

  std::stringstream ss_ostram;
  ss_ostram << program;

  EXPECT_EQ(ss.str(), ss_ostram.str());
}

TEST(program_test, slice_combine_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  // (2) Create an empty program object
  pir::Program program(ctx);
  //   pir::Program *program = new pir::Program();
  EXPECT_EQ(program.block()->empty(), true);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);

  // (4) Def a = ParameterOp("a")
  std::string op1_name = pir::ParameterOp::name();
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"parameter_name", pir::StrAttribute::get(ctx, "a")}};
  pir::Operation *op1 =
      pir::Operation::Create({}, op1_attribute, {fp32_dtype}, op1_info);
  program.block()->push_back(op1);

  // (5) Def b = Constant("b")
  std::string op2_name = std::string(pir::ConstantOp::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  pir::AttributeMap attr_map;
  attr_map.insert(std::pair<std::string, pir::Attribute>(
      "value", pir::FloatAttribute::get(ctx, 2.0)));
  pir::Operation *op2 =
      pir::Operation::Create({}, attr_map, {fp32_dtype}, op2_info);
  program.block()->push_back(op2);

  // (6) Def combine_op = CombineOp("a", "b")
  std::string combine_op_name = std::string(pir::CombineOp::name());
  pir::OpInfo combine_op_info = ctx->GetRegisteredOpInfo(combine_op_name);
  pir::Type output_type = pir::VectorType::get(
      ctx, std::vector<pir::Type>({fp32_dtype, fp32_dtype}));
  pir::Operation *combine_op = pir::Operation::Create(
      {op1->result(0), op2->result(0)}, {}, {output_type}, combine_op_info);
  pir::CombineOp combine_op_type = combine_op->dyn_cast<pir::CombineOp>();
  EXPECT_TRUE(combine_op_type.out());
  program.block()->push_back(combine_op);

  // (7) Def slice_op = SliceOp(combine_op, 0)
  std::string slice_op_name = std::string(pir::SliceOp::name());
  pir::OpInfo slice_op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  pir::Attribute index_attr = pir::Int32Attribute::get(ctx, 0);
  pir::Operation *slice_op = pir::Operation::Create({combine_op->result(0)},
                                                    {{"index", index_attr}},
                                                    {fp32_dtype},
                                                    slice_op_info);
  program.block()->push_back(slice_op);

  // (8) Traverse Program
  EXPECT_EQ(program.block()->size() == 4, true);
}

TEST(program_test, builder) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::Type full_op_output = full_op->result_type(0);
  EXPECT_EQ(program.block()->size(), 1u);
  EXPECT_EQ(program.block()->back(), *full_op.operation());
  EXPECT_EQ(full_op.num_operands(), 0u);
  EXPECT_EQ(full_op.num_results(), 1u);
  EXPECT_EQ(full_op.attributes().size(), 5u);
  EXPECT_EQ(
      full_op_output.dyn_cast<paddle::dialect::DenseTensorType>().offset() == 0,
      true);
  for (auto dim : common::vectorize(
           full_op_output.dyn_cast<paddle::dialect::DenseTensorType>()
               .dims())) {
    EXPECT_EQ(dim == 2, true);
  }

  pir::ConstantOp constant = builder.Build<pir::ConstantOp>(
      pir::Int32Attribute::get(ctx, 2), pir::Int32Type::get(ctx));
  EXPECT_EQ(program.block()->size() == 2, true);
  EXPECT_EQ(constant.value().dyn_cast<pir::Int32Attribute>().data() == 2, true);
}
