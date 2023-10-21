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

#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

namespace {

void AppendAttrForReduceOp(const ::pir::Operation& op,
                           utils::AttributeMap& attrs) {  // NOLINT
  auto* source_op =
      op.operand_source(/*dim_idx=*/1).dyn_cast<::pir::OpResult>().owner();
  CHECK(source_op->isa<paddle::dialect::FullIntArrayOp>());
  const std::vector<int64_t>& dim_val =
      source_op->attributes()
          .at("value")
          .dyn_cast<paddle::dialect::IntArrayAttribute>()
          .data()
          .GetData();
  std::vector<int> dim(dim_val.begin(), dim_val.end());
  attrs["dim"] = dim;
}

}  // namespace

#define REGISTER_OPERAND_RULE(OP, args...)                                    \
  operand_funcs_[paddle::dialect::OP::name()] = []() -> std::vector<size_t> { \
    return {args};                                                            \
  };

#define REGISTER_ATTR_RULE(OP, func) \
  attr_funcs_[paddle::dialect::OP::name()] = func;

void OpMapper::RegisterMapRules() {
  // max(x, dim) -> reduce_max(x)
  REGISTER_OPERAND_RULE(MaxOp, 0);
  REGISTER_OPERAND_RULE(SumOp, 0);
  REGISTER_OPERAND_RULE(MinOp, 0);
  REGISTER_OPERAND_RULE(ProdOp, 0);
  REGISTER_ATTR_RULE(MaxOp, AppendAttrForReduceOp);
  REGISTER_ATTR_RULE(SumOp, AppendAttrForReduceOp);
  REGISTER_ATTR_RULE(MinOp, AppendAttrForReduceOp);
  REGISTER_ATTR_RULE(ProdOp, AppendAttrForReduceOp);
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
