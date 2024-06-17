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

#include <glog/logging.h>

#include "paddle/common/macros.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_instrumentation.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/utils/string/pretty_log.h"

REGISTER_FILE_SYMBOLS(print_statistics);

namespace pir {

class PrintStatistics : public PassInstrumentation {
 public:
  PrintStatistics() = default;

  ~PrintStatistics() override = default;

  void RunBeforePass(Pass *pass, Operation *op) override {
    paddle::string::PrettyLogH1("--- Running PIR pass [%s]",
                                pass->pass_info().name);
  }

  void RunAfterPass(Pass *pass, Operation *op) override {
    if (pass->Has("__match_count_1__") && pass->Has("__match_count_2__")) {
      auto match_count_1 = pass->Get<int64_t>("__match_count_1__");
      auto match_count_2 = pass->Get<int64_t>("__match_count_2__");
      if (match_count_1 > 0 || match_count_2 > 0) {
        LOG(INFO) << "--- detected [" << match_count_1 << ", " << match_count_2
                  << "] subgraphs!";
      }
    } else if (pass->Has("__match_count__")) {
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
