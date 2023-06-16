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

#include "paddle/fluid/ir/dialect/kernel_dialect.h"
#include "paddle/fluid/ir/dialect/kernel_op.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

TEST(program_test, program) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  auto kernel_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::PaddleKernelDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  // (2) Create an empty program object
  ir::Program program(ctx);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  phi::Place place(phi::AllocationType::CPU);
  ir::Type fp32_dtype = ir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  std::string op1_name = paddle::dialect::PhiKernelOp::name();

  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);

  std::unordered_map<std::string, ir::Attribute> op1_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "a")}};

  auto allocated_dense_tensor_dtype =
      paddle::dialect::AllocatedDenseTensorType::get(
          ctx, place, fp32_dtype, dims, data_layout, lod, offset);
  std::stringstream ss;
  kernel_dialect->PrintType(allocated_dense_tensor_dtype, ss);
  ASSERT_EQ(ss.str() == "cpu_tensor<2x2xf32>", true);
  ASSERT_EQ(allocated_dense_tensor_dtype.place() == place, true);
  ASSERT_EQ(allocated_dense_tensor_dtype.dims() == dims, true);
  ASSERT_EQ(allocated_dense_tensor_dtype.data_layout() == data_layout, true);
  ASSERT_EQ(allocated_dense_tensor_dtype.lod() == lod, true);
  ASSERT_EQ(allocated_dense_tensor_dtype.offset() == 0, true);

  ir::Operation *op1 = ir::Operation::Create(
      {}, op1_attribute, {allocated_dense_tensor_dtype}, op1_info);

  ASSERT_EQ(op1 != nullptr, true);
}
