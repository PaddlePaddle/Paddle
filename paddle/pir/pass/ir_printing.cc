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

#include <ostream>
#include <string>
#include <unordered_map>

#include "paddle/pir/core/operation.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_instrumentation.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/utils.h"

namespace pir {

namespace {
void PrintIR(Operation *op, bool print_module, std::ostream &os) {
  if (!print_module) {
    op->Print(os << "\n");
    return;
  }

  // Find the top-level operation.
  auto *top_op = op;
  while (auto *parent_op = top_op->GetParentOp()) {
    top_op = parent_op;
  }
  top_op->Print(os);
}
}  // namespace

class IRPrinting : public PassInstrumentation {
 public:
  explicit IRPrinting(std::unique_ptr<PassManager::IRPrinterOption> option)
      : option_(std::move(option)) {}

  ~IRPrinting() override = default;

  void RunBeforePass(Pass *pass, Operation *op) override {
    if (option_->print_on_change()) {
      // TODO(liuyuanle): support print on change
    }

    option_->PrintBeforeIfEnabled(pass, op, [&](std::ostream &os) {
      std::string header =
          "IRPrinting on " + op->name() + " before " + pass->name() + " pass";
      detail::PrintHeader(header, os);
      PrintIR(op, option_->print_module(), os);
      os << "\n\n";
    });
  }

  void RunAfterPass(Pass *pass, Operation *op) override {
    if (option_->print_on_change()) {
      // TODO(liuyuanle): support print on change
    }

    option_->PrintAfterIfEnabled(pass, op, [&](std::ostream &os) {
      std::string header =
          "IRPrinting on " + op->name() + " after " + pass->name() + " pass";
      detail::PrintHeader(header, os);
      PrintIR(op, option_->print_module(), os);
      os << "\n\n";
    });
  }

 private:
  std::unique_ptr<PassManager::IRPrinterOption> option_;

  // TODO(liuyuanle): Add IRFingerPrint to support print on change.
};

void PassManager::EnableIRPrinting(std::unique_ptr<IRPrinterOption> option) {
  AddInstrumentation(std::make_unique<IRPrinting>(std::move(option)));
}

}  // namespace pir
