// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/tensorrt/trt_op_teller_pass.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/ir/basic_kernels.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"

namespace infrt {
namespace trt {
// Implementation of the trtOpTellerPass。
void TRTOpTellerPass::runOnFunction() {
  mlir::Block &body = getFunction().front();
  std::vector<mlir::Operation *> worklist;
  worklist.reserve(body.getOperations().size());
  for (auto &op : body) {
    worklist.push_back(&op);
  }
  // Build GraphOp.
  mlir::OpBuilder builder(&body, body.begin());
  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    if (op == nullptr) continue;
    if (op->getName().getStringRef().substr(0, 3) != "pd.") continue;
    if (::llvm::dyn_cast_or_null<infrt::pd::FeedOp>(op)) continue;
    if (::llvm::dyn_cast_or_null<infrt::pd::FetchOp>(op)) continue;
    if (::llvm::dyn_cast_or_null<infrt::pd::GraphOp>(op)) continue;
    if (::llvm::dyn_cast_or_null<::infrt::ReturnOp>(op)) continue;

    builder.setInsertionPoint(op);
    auto loc = getFunction().getLoc();
    auto graph_op = builder.create<infrt::pd::GraphOp>(
        loc, op->getResultTypes(), op->getOperands());

    ::llvm::SmallVector<mlir::Value, 4> tblgen_repl_values;
    for (auto v :
         ::llvm::SmallVector<mlir::Value, 4>{graph_op.getODSResults(0)}) {
      tblgen_repl_values.push_back(v);
    }
    op->replaceAllUsesWith(tblgen_repl_values);
    // Build graph op.
    mlir::Block *block = new mlir::Block;
    graph_op.body().push_back(block);
    op->moveBefore(block, block->begin());
    builder.setInsertionPointToEnd(block);
    builder.create<::infrt::ReturnOp>(loc, op->getResults());
  }
}
}  // namespace trt
}  // namespace infrt
