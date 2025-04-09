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

#include <string>

#include "paddle/cinn/pass/pass.h"
#include "paddle/cinn/pass/pass_adaptor.h"

namespace cinn {
namespace optim {

template <typename PassT, typename PassAdaptorT>
class PassManager {
 public:
  virtual LogicalResult Run(ir::LoweredFunc func) {
    return adaptor_.RunPipeline(func, passes_);
  }
  virtual LogicalResult Run(ir::stmt::BlockRef block) {
    return adaptor_.RunPipeline(block, passes_);
  }
  void AddPass(std::unique_ptr<PassT> pass) {
    passes_.emplace_back(std::move(pass));
  }

 private:
  std::vector<std::unique_ptr<PassT>> passes_;
  PassAdaptorT adaptor_;
};

using FuncPassManager = PassManager<FuncPass, detail::FuncPassAdaptor>;
using BlockPassManager = PassManager<BlockPass, detail::BlockPassAdaptor>;
using StmtPassManager = PassManager<StmtPass, detail::StmtPassAdaptor>;
using ExprPassManager = PassManager<ExprPass, detail::ExprPassAdaptor>;

}  // namespace optim
}  // namespace cinn
