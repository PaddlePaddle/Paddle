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

#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/common/macros.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_instrumentation.h"
#include "paddle/pir/include/pass/pass_manager.h"

REGISTER_FILE_SYMBOLS(pass_timing);

namespace pir {
namespace {
class Timer {
 public:
  Timer() = default;

  ~Timer() = default;

  void Start() { start_time_ = std::chrono::steady_clock::now(); }

  void Stop() { walk_time += std::chrono::steady_clock::now() - start_time_; }

  double GetTimePerSecond() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(walk_time)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;

  std::chrono::nanoseconds walk_time = std::chrono::nanoseconds(0);
};
}  // namespace

class PassTimer : public PassInstrumentation {
 public:
  explicit PassTimer(bool print_module) : print_module_(print_module) {}
  ~PassTimer() override = default;

  void RunBeforePipeline(pir::Operation* op) override {
    pipeline_timers_[op] = Timer();
    pipeline_timers_[op].Start();
  }

  void RunAfterPipeline(Operation* op) override {
    pipeline_timers_[op].Stop();
    std::ostringstream oss;
    PrintTime(op, oss);
    std::cout << oss.str() << std::endl;
  }

  void RunBeforePass(Pass* pass, Operation* op) override {
    if (!pass_timers_.count(op)) {
      pass_timers_[op] = {};
    }
    pass_timers_[op][pass->name()] = Timer();
    pass_timers_[op][pass->name()].Start();
  }

  void RunAfterPass(Pass* pass, Operation* op) override {
    pass_timers_[op][pass->name()].Stop();
  }

 private:
  void PrintTime(Operation* op, std::ostream& os) {
    if (print_module_ && op->name() != "builtin.module") return;

    std::string header = "PassTiming on " + op->name();
    detail::PrintHeader(header, os);

    os << "  Total Execution Time: " << std::fixed << std::setprecision(3)
       << pipeline_timers_[op].GetTimePerSecond() << " seconds\n\n";
    os << "  ----Walk Time----  ----Name----\n";

    auto& map = pass_timers_[op];
    std::vector<std::pair<std::string, Timer>> pairs(map.begin(), map.end());
    std::sort(pairs.begin(),
              pairs.end(),
              [](const std::pair<std::string, Timer>& lhs,
                 const std::pair<std::string, Timer>& rhs) {
                return lhs.second.GetTimePerSecond() >
                       rhs.second.GetTimePerSecond();
              });

    for (auto& v : pairs) {
      os << "  " << std::fixed << std::setw(8) << std::setprecision(3)
         << v.second.GetTimePerSecond() << " (" << std::setw(5)
         << std::setprecision(1)
         << 100 * v.second.GetTimePerSecond() /
                pipeline_timers_[op].GetTimePerSecond()
         << "%)"
         << "  " << v.first << "\n";
    }
  }

 private:
  bool print_module_;

  std::unordered_map<Operation*, Timer> pipeline_timers_;

  std::unordered_map<Operation*,
                     std::unordered_map<std::string /*pass name*/, Timer>>
      pass_timers_;
};

void PassManager::EnablePassTiming(bool print_module) {
  AddInstrumentation(std::make_unique<PassTimer>(print_module));
}

}  // namespace pir
