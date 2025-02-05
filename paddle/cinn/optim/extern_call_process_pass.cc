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

#include "paddle/cinn/optim/extern_call_process_pass.h"

namespace cinn {
namespace optim {
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::StmtRef;
using ir::stmt::Store;

namespace {

void ProcessMultiOutputStore(BlockRef block) {
  const auto& stmts = block->stmts();
  std::vector<StmtRef> new_stmts;

  for (const auto& stmt : stmts) {
    if (stmt.isa<Store>()) {
      const auto& store_op = stmt.as<Store>()->value();
      const auto& call = store_op.As<ir::Call>();
      if (call && call->is_extern_call() && !call->write_args.empty()) {
        new_stmts.emplace_back(Evaluate(store_op));
      } else {
        new_stmts.emplace_back(stmt);
      }
    } else {
      new_stmts.emplace_back(stmt);
    }
  }

  block->set_stmts(new_stmts);
}

}  // namespace

LogicalResult ExternCallMultiOutputShallowStorePass::Run(BlockRef block) {
  ProcessMultiOutputStore(block);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateExternCallMultiOutputShallowStorePass() {
  return std::make_unique<ExternCallMultiOutputShallowStorePass>();
}

}  // namespace optim
}  // namespace cinn
