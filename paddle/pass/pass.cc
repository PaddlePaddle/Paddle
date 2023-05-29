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

#include "paddle/pass/pass.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/operation.h"
#include "paddle/pass/pass_adaptor.h"
#include "paddle/pass/pass_manager.h"

namespace ir {

void detail::PassAdaptor::Run(ir::Operation* op, uint8_t opt_level) {
  RunImpl(op, opt_level);
}

void detail::PassAdaptor::RunImpl(ir::Operation* op, uint8_t opt_level) {
  // TODO(liuyuanle): Support block, region, etc.
  return;
}

bool detail::PassAdaptor::RunPipeline(const PassManager& pm,
                                      ir::Operation* op,
                                      uint8_t opt_level) {
  for (auto& pass : pm.GetPasses()) {
    if (pass->CanScheduleOn(op)) {
      if (!RunPass(pass.get(), op, opt_level)) {
        return false;
      }
    }
  }

  // Apply pass manager on all nested ir.
  if (!RunPass(pm.pass_adaptor_.get(), op, opt_level)) {
    return false;
  }

  return true;
}

bool detail::PassAdaptor::RunPass(Pass* pass,
                                  ir::Operation* op,
                                  uint8_t opt_level) {
  if (opt_level < pass->info_.opt_level) return true;

  pass->pass_state_ = detail::PassExecutionState(op);

  if (auto* adaptor = dynamic_cast<detail::PassAdaptor*>(pass)) {
    adaptor->Run(op, opt_level);
  } else {
    pass->Run(op);
  }

  bool pass_failed = pass->pass_state_->pass_failed;

  return !pass_failed;
}

PassManager::PassManager(ir::IrContext* context, uint8_t opt_level)
    : context_(context), opt_level_(opt_level) {
  pass_adaptor_ = std::make_unique<detail::PassAdaptor>(this);
}

bool PassManager::Run(ir::Operation* op) {
  if (!Initialize(context_)) {
    return false;
  }
  return RunPasses(op);
}

bool PassManager::RunPasses(ir::Operation* op) {
  return detail::PassAdaptor::RunPipeline(*this, op, opt_level_);
}

bool PassManager::Initialize(ir::IrContext* context) {
  for (auto& pass : GetPasses()) {
    if (!pass->Initialize(context)) return false;
  }

  return true;
}

}  // namespace ir
