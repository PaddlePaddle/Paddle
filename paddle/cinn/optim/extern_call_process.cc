// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/extern_call_process.h"

#include "paddle/cinn/ir/utils/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct ExternCallMultiOutputShallowStoreMutator : public ir::IRMutator<> {
  void operator()(Expr* e) { ir::IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* call = op->value.As<ir::Call>();
    if (call && call->is_extern_call() && !call->write_args.empty()) {
      *expr = op->value;
    }
  }
};

}  // namespace

void ExternCallMultiOutputShallowStore(Expr* e) {
  ExternCallMultiOutputShallowStoreMutator()(e);
}

}  // namespace optim
}  // namespace cinn
