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
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/type.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"

TEST(shapedtype_test, shapedtype_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType shaped_type = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  EXPECT_EQ(shaped_type.dtype().isa<pir::Float32Type>(), true);
  EXPECT_EQ(shaped_type.dims(), dims);
  EXPECT_EQ(shaped_type.data_layout(), data_layout);
  EXPECT_EQ(shaped_type.lod(), lod);
  EXPECT_EQ(shaped_type.offset(), offset);

  pir::ShapedTypeInterface interface =
      shaped_type.dyn_cast_interface<pir::ShapedTypeInterface>();

  EXPECT_EQ(interface.getElementType().isa<pir::Float32Type>(), true);
  EXPECT_EQ(interface.getShape(), dims);
  EXPECT_EQ(interface.kDynamic, std::numeric_limits<int64_t>::min());
  EXPECT_EQ(interface.getRank(), 2);
  EXPECT_EQ(interface.isDynamic(2), false);
  EXPECT_EQ(interface.isDynamicShape(dims), false);
  EXPECT_EQ(interface.isDynamicDim(1), false);
  EXPECT_EQ(interface.getNumDynamicDims(), 0);
  EXPECT_EQ(interface.getDimSize(0), 2);
}
