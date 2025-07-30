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

#pragma once

#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {
namespace detail {

template <typename PassT>
class PassAdaptor {
 public:
  LogicalResult RunPipeline(ir::LoweredFunc func,
                            const std::vector<std::unique_ptr<PassT>>& passes) {
    // TODO(Hongqing-work): Add instrumentation and AnalysisManager. Remove stmt
    // convert after update all the backend passes.
    func->body_block = ir::ConvertExprBlockToStmtBlock(func->body);
    LogicalResult res = Run(func, passes);
    func->body = ir::ConvertStmtBlockToExprBlock(func->body_block);
    return res;
  }

  LogicalResult RunPipeline(ir::stmt::BlockRef block,
                            const std::vector<std::unique_ptr<PassT>>& passes) {
    LogicalResult res = Run(block, passes);
    return res;
  }

 protected:
  virtual LogicalResult Run(
      ir::LoweredFunc func,
      const std::vector<std::unique_ptr<PassT>>& passes) = 0;
  virtual LogicalResult Run(
      ir::stmt::BlockRef block,
      const std::vector<std::unique_ptr<PassT>>& passes) = 0;
};

class FuncPassAdaptor : public PassAdaptor<FuncPass> {
 private:
  LogicalResult Run(
      ir::LoweredFunc func,
      const std::vector<std::unique_ptr<FuncPass>>& passes) override;
  LogicalResult Run(
      ir::stmt::BlockRef block,
      const std::vector<std::unique_ptr<FuncPass>>& passes) override;
};

class BlockPassAdaptor : public PassAdaptor<BlockPass> {
 private:
  LogicalResult Run(
      ir::LoweredFunc func,
      const std::vector<std::unique_ptr<BlockPass>>& passes) override;
  LogicalResult Run(
      ir::stmt::BlockRef block,
      const std::vector<std::unique_ptr<BlockPass>>& passes) override;
};

class ExprPassAdaptor;

class StmtPassAdaptor : public PassAdaptor<StmtPass> {
  friend class ExprPassAdaptor;

 private:
  LogicalResult Run(
      ir::LoweredFunc func,
      const std::vector<std::unique_ptr<StmtPass>>& passes) override;
  LogicalResult Run(
      ir::stmt::BlockRef block,
      const std::vector<std::unique_ptr<StmtPass>>& passes) override;
};

class ExprPassAdaptor : public PassAdaptor<ExprPass> {
 private:
  LogicalResult Run(
      ir::LoweredFunc func,
      const std::vector<std::unique_ptr<ExprPass>>& passes) override;
  LogicalResult Run(
      ir::stmt::BlockRef block,
      const std::vector<std::unique_ptr<ExprPass>>& passes) override;
};

}  // namespace detail
}  // namespace optim
}  // namespace cinn
