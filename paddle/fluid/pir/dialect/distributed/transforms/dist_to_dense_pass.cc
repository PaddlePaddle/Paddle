// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/distributed/transforms/dist_to_dense_pass.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_interface.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"

using paddle::dialect::DistDenseTensorType;

COMMON_DECLARE_bool(print_ir);

namespace paddle {
namespace dialect {

pir::Type CastToLocalType(pir::Type type) {
  if (auto dist_type = type.dyn_cast<DistTypeInterface>()) {
    return dist_type.local_type();
  } else if (auto vec_type = type.dyn_cast<pir::VectorType>()) {
    std::vector<pir::Type> local_types;
    for (size_t i = 0; i < vec_type.size(); ++i) {
      local_types.push_back(CastToLocalType(vec_type[i]));
    }
    return pir::VectorType::get(vec_type.ir_context(), local_types);
  } else if (!type) {
    // skip if <<NULL TYPE>>
    return nullptr;
  } else {
    // TODO(2024-Q2) not all value are dist type
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "The type[%s] is not Dist type.", type));
  }
}

inline bool IsDistType(pir::Type type) { return type.isa<DistTypeInterface>(); }

void ProcessDistBlock(pir::Block* block) {
  auto ctx = pir::IrContext::Instance();
  for (auto& val : *block) {
    pir::Operation* op_item = &val;
    VLOG(6) << "dist_to_dense main loop over op [" << op_item->name() << "].";

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto result = op_item->result(i);
      result.set_type(CastToLocalType(result.type()));
    }
    if (op_item->isa<DataOp>()) {
      auto dense_tensor_type =
          op_item->result(0).type().dyn_cast<pir::DenseTensorType>();
      auto shape = common::vectorize(dense_tensor_type.dims());
      pir::Attribute attr_shape = IntArrayAttribute::get(
          pir::IrContext::Instance(), phi::IntArray(shape));
      op_item->set_attribute("shape", attr_shape);
    } else if (op_item->isa<ReshapeOp>()) {
      auto local_dims =
          op_item->result_type(0).dyn_cast<pir::DenseTensorType>().dims();
      auto shape_value = op_item->operand_source(1);
      auto prev_op = shape_value.defining_op();
      if (prev_op == nullptr || !(prev_op->isa<FullIntArrayOp>())) {
        auto op_name = prev_op ? prev_op->name() : "nullptr";
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "The reshape op's shape input mush be the result of  "
            "FullIntArrayOp. but it is %s",
            op_name));
      }
      auto array_attr = prev_op->attribute<pir::ArrayAttribute>("value");
      PADDLE_ENFORCE_EQ(array_attr.size(),
                        local_dims.size(),
                        phi::errors::PreconditionNotMet(
                            "The reshape's shape inputs element's size must "
                            "equal to result's dim size."));
      std::vector<pir::Attribute> new_dims;
      for (int index = 0; index < local_dims.size(); ++index) {
        new_dims.push_back(pir::Int64Attribute::get(ctx, local_dims[index]));
      }
      prev_op->set_attribute("value", pir::ArrayAttribute::get(ctx, new_dims));
    }
    // TODO(2024-Q2) not all op are dist type
    // PADDLE_ENFORCE_EQ(
    //     (op_item->HasAttribute(kAttrOpDistAttr) &&
    //      op_item->attribute(kAttrOpDistAttr)
    //          .isa<paddle::dialect::OperationDistAttribute>()),
    //     true,
    //     common::errors::PreconditionNotMet("The op [%s] has not
    //     op_dist_attr.",
    //                                        op_item->name()));
    if (op_item->HasAttribute(kAttrOpDistAttr)) {
      op_item->erase_attribute(kAttrOpDistAttr);
    }

    // TODO(2024-Q2) Handle other special dist op in future.
  }
}

/* Verification:
    1. no operator has not OperatorDistAttr.
    2. all Values (Results) are DenseTensorType.
    3. no shard_tensor / reshard in block.
*/
void VerifyDenseBlock(pir::Block* block) {
  for (auto& val : *block) {
    pir::Operation* op_item = &val;

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto result = op_item->result(i);

      PADDLE_ENFORCE_EQ(
          IsDistType(result.type()),
          false,
          phi::errors::PreconditionNotMet(
              "Block op [%s] still contain dist type.", op_item->name()));
    }

    PADDLE_ENFORCE_EQ(
        op_item->HasAttribute(kAttrOpDistAttr),
        false,
        common::errors::PreconditionNotMet(
            "The op [%s] still has op_dist_attr.", op_item->name()));
  }
}

void DistToDensePass(pir::Program* prog) {
  if (FLAGS_print_ir) {
    VLOG(0) << "IR before DistToDense Pass = " << *prog;
  }

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessDistBlock(prog->block());
  VerifyDenseBlock(prog->block());

  if (FLAGS_print_ir) {
    VLOG(0) << "IR after DistToDense Pass = " << *prog;
  }
}

}  // namespace dialect
}  // namespace paddle
