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

#include "paddle/fluid/paddle_dialect/dialect.h"
#include "paddle/fluid/paddle_dialect/type.h"
#include "paddle/fluid/paddle_dialect/utils.h"
#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/program.h"
#include "paddle/ir/utils.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

class AddOp : public ir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "Add"; }
  static const char **attributes_name_;
  static uint32_t attributes_num() { return 0; }
};
const char **AddOp::attributes_name_ = nullptr;

TEST(program_test, program) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  builtin_dialect->RegisterOp<AddOp>();
  ir::Dialect *paddle_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  // (2) Create an empty program object
  ir::Program *program = new ir::Program();
  EXPECT_EQ(program->ops().size() == 0, true);
  EXPECT_EQ(program->parameters().size() == 0, true);

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
  ir::Parameter *parameter_a =
      new ir::Parameter(reinterpret_cast<void *>(data_a.data()),
                        4 * sizeof(float),
                        dense_tensor_dtype);
  program->SetParameter("a", parameter_a);

  std::vector<float> data_b = {5, 6, 7, 8};
  ir::Parameter *parameter_b =
      new ir::Parameter(reinterpret_cast<void *>(data_b.data()),
                        4 * sizeof(float),
                        dense_tensor_dtype);
  program->SetParameter("b", parameter_b);
  EXPECT_EQ(program->parameters().size() == 2, true);

  // (4) Def a = GetParameterOp("a"), and create DenseTensor for a.
  std::string op1_name =
      builtin_dialect->name() + "." + std::string(ir::GetParameterOp::name());
  ir::OpInfoImpl *op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::map<ir::StrAttribute, ir::Attribute> op1_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "a")}};
  ir::DictionaryAttribute op1_attribute =
      ir::DictionaryAttribute::get(ctx, op1_attribute_map);
  ir::Operation *op1 = ir::Operation::create(
      {}, {dense_tensor_dtype}, op1_attribute, op1_info, program);

  EXPECT_EQ(op1->GetResultByIndex(0).type().dialect().id(),
            paddle_dialect->id());
  using Interface = paddle::dialect::ParameterConvertInterface;
  Interface *a_interface = op1->GetResultByIndex(0)
                               .type()
                               .dialect()
                               .GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> a_var =
      a_interface->ParameterToVariable(program->GetParameter("a"));
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
  ir::OpInfoImpl *op2_info = ctx->GetRegisteredOpInfo(op2_name);
  std::map<ir::StrAttribute, ir::Attribute> op2_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "b")}};
  ir::DictionaryAttribute op2_attribute =
      ir::DictionaryAttribute::get(ctx, op2_attribute_map);
  ir::Operation *op2 = ir::Operation::create(
      {}, {dense_tensor_dtype}, op2_attribute, op2_info, program);

  EXPECT_EQ(op2->GetResultByIndex(0).type().dialect().id(),
            paddle_dialect->id());
  Interface *b_interface = op2->GetResultByIndex(0)
                               .type()
                               .dialect()
                               .GetRegisteredInterface<Interface>();
  std::shared_ptr<paddle::framework::Variable> b_var =
      b_interface->ParameterToVariable(program->GetParameter("b"));
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
  ir::OpInfoImpl *op3_info = ctx->GetRegisteredOpInfo(op3_name);
  ir::Operation *op3 = ir::Operation::create(
      {op1->GetResultByIndex(0), op2->GetResultByIndex(0)},
      {dense_tensor_dtype},
      nullptr,
      op3_info,
      program);

  //   paddle::platform::DeviceContext* dev_ctx =
  //         paddle::platform::DeviceContextPool::Instance().Get(
  //             paddle::platform::CPUPlace());
  //   phi::DenseTensor c_tensor = phi::Add<float,
  //   paddle::platform::DeviceContext>(*dev_ctx, a_tensor, b_tensor);
  //   EXPECT_EQ(c_tensor.numel(), 4);

  // (7) Def SetParameterOp(c, "c")
  std::string op4_name =
      builtin_dialect->name() + "." + std::string(ir::SetParameterOp::name());
  ir::OpInfoImpl *op4_info = ctx->GetRegisteredOpInfo(op4_name);
  std::map<ir::StrAttribute, ir::Attribute> op4_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "c")}};
  ir::DictionaryAttribute op4_attribute =
      ir::DictionaryAttribute::get(ctx, op4_attribute_map);
  ir::Operation *op4 = ir::Operation::create(
      {op3->GetResultByIndex(0)}, {}, op4_attribute, op4_info, program);
  std::cout << op4 << std::endl;

  // (4) Traverse Program
  std::list<ir::Operation *> ops = program->ops();
  EXPECT_EQ(ops.size() == 4, true);
}
