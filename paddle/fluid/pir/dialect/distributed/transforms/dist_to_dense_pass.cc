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

using paddle::dialect::DistDenseTensorType;

COMMON_DECLARE_bool(print_ir);

namespace paddle {
namespace dialect {

inline pir::Type CastToLocalType(pir::Type dist_type) {
  return dist_type.dyn_cast<DistTypeInterface>().local_type();
}

inline bool IsDistType(pir::Type type) { return type.isa<DistTypeInterface>(); }

void ProcessDistBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(6) << "dist_to_dense main loop over op [" << op_item->name() << "].";

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto result = op_item->result(i);
      auto origin_type = result.type();
      if (IsDistType(origin_type)) {
        auto local_type = CastToLocalType(origin_type);
        result.set_type(local_type);
      } else if (origin_type) {  // skip if <<NULL TYPE>>
        // TODO(2024-Q2) not all value are dist type
        PADDLE_THROW(platform::errors::PreconditionNotMet(
            "The op [%s]'s [%d]th result is not Dist type.",
            op_item->name(),
            i));
      }
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
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);

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

std::shared_ptr<pir::Program> DistToDensePass(pir::Program* prog) {
  if (FLAGS_print_ir) {
    VLOG(0) << "IR before DistToDense Pass = " << *prog;
  }

  pir::IrMapping mapper;
  auto new_prog = prog->Clone(mapper);

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessDistBlock(new_prog->block());
  VLOG(6) << "IR before VerifyDenseBlock Pass = " << *new_prog;
  VerifyDenseBlock(new_prog->block());

  if (FLAGS_print_ir) {
    VLOG(0) << "IR after DistToDense Pass = " << *new_prog;
  }

  return new_prog;
}

}  // namespace dialect
}  // namespace paddle
