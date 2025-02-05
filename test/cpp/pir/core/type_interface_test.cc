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
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/type.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"

TEST(shape_dtype_test, shape_dtype_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  EXPECT_EQ(dense_tensor_type.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(dense_tensor_type.dims(), dims);
  EXPECT_EQ(dense_tensor_type.data_layout(), data_layout);
  EXPECT_EQ(dense_tensor_type.lod(), lod);
  EXPECT_EQ(dense_tensor_type.offset(), offset);

  pir::ShapedTypeInterface dense_tensor_type_interface =
      dense_tensor_type.dyn_cast<pir::ShapedTypeInterface>();

  EXPECT_TRUE(dense_tensor_type_interface);
  EXPECT_EQ(
      dense_tensor_type_interface.GetElementType().isa<pir::Float32Type>(),
      true);
  EXPECT_EQ(dense_tensor_type_interface.GetShape(), dims);
  EXPECT_EQ(dense_tensor_type_interface.kDynamic, std::int64_t(-1));
  EXPECT_EQ(dense_tensor_type_interface.GetRank(), 2);
  EXPECT_EQ(dense_tensor_type_interface.IsDynamic(2), false);
  EXPECT_EQ(dense_tensor_type_interface.IsDynamicShape(), false);
  EXPECT_EQ(dense_tensor_type_interface.IsDynamicDim(1), false);
  EXPECT_EQ(dense_tensor_type_interface.GetNumDynamicDims(), 0);
  EXPECT_EQ(dense_tensor_type_interface.GetDimSize(0), 2);

  pir::Type fp32_type = pir::Float32Type::get(ctx);
  pir::ShapedTypeInterface fp32_type_interface =
      fp32_type.dyn_cast<pir::ShapedTypeInterface>();

  EXPECT_FALSE(fp32_type_interface);
}
