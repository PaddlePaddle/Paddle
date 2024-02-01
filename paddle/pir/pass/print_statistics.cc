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

#include "paddle/common/macros.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_instrumentation.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/utils/string/pretty_log.h"

REGISTER_FILE_SYMBOLS(print_statistics);

namespace pir {

class PrintStatistics : public PassInstrumentation {
 public:
  PrintStatistics() = default;

  ~PrintStatistics() override = default;

  void RunBeforePass(Pass *pass, Operation *op) override {
    if (pass->name() == "replace_fetch_with_shadow_output_pass") {
      return;
    }
    paddle::string::PrettyLogH1("--- Running PIR pass [%s]",
                                pass->pass_info().name);
  }

  void RunAfterPass(Pass *pass, Operation *op) override {
    if (pass->name() == "replace_fetch_with_shadow_output_pass") {
      return;
    }
    if (pass->Has("__match_count__") && pass->Has("__all_count__")) {
      auto match_count = pass->Get<int64_t>("__match_count__");
      auto all_count = pass->Get<int64_t>("__all_count__");
      IR_ENFORCE(match_count <= all_count,
                 "match_count: %d should smaller than all_count: %d",
                 match_count,
                 all_count);
      if (match_count > 0) {
        LOG(INFO) << "--- detected [" << match_count << "/" << all_count
                  << "] subgraphs!";
      }
    } else if (pass->Has("__match_count__") && !pass->Has("__all_count__")) {
      auto match_count = pass->Get<int64_t>("__match_count__");
      if (match_count > 0) {
        LOG(INFO) << "--- detected [" << match_count << "] subgraphs!";
      }
    } else if (pass->Has("__custom_log__")) {
      auto custom_log = pass->Get<std::string>("__custom_log__");
      if (!custom_log.empty()) {
        LOG(INFO) << custom_log;
      }
    }
  }
};

void PassManager::EnablePrintStatistics() {
  AddInstrumentation(std::make_unique<PrintStatistics>());
}

}  // namespace pir
