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

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <thread>

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include "Pass/PassInstrumentation.h"
#include "Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
namespace infra {

namespace {

class Timer {
 public:
  Timer() = default;

  void Start() { start_time_ = std::chrono::steady_clock::now(); }

  void Stop() {
    auto new_time = std::chrono::steady_clock::now() - start_time_;
    wall_time += new_time;
  }

  double GetTimePerSecond() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(wall_time)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::chrono::nanoseconds wall_time = std::chrono::nanoseconds(0);
};

struct PassTiming : public PassInstrumentation {
  std::unordered_map<void*, Timer> pipeline_timers_;
  std::unordered_map<void* /*operation*/,
                     std::unordered_map<std::string /*pass name*/, Timer>>
      pass_times_;

  void RunBeforePipeline(mlir::Operation* op) override {
    pipeline_timers_[op] = Timer();
    pipeline_timers_[op].Start();
  }

  void RunAfterPipeline(mlir::Operation* op) override {
    pipeline_timers_[op].Stop();

    std::ostringstream os;
    PrintTime(os, op);
    llvm::outs() << os.str() << "\n";
  }

  void RunBeforePass(Pass* pass, mlir::Operation* op) override {
    if (!pass_times_.count(op)) {
      pass_times_[op] = {};
    }
    pass_times_[op][pass->GetPassInfo().name] = Timer();
    pass_times_[op][pass->GetPassInfo().name].Start();
  }

  void RunAfterPass(Pass* pass, mlir::Operation* op) override {
    pass_times_[op][pass->GetPassInfo().name].Stop();
  }

  void PrintTime(std::ostringstream& os, mlir::Operation* op) {
    std::string header = "PassTiming on " + op->getName().getStringRef().str();
    unsigned padding = (80 - header.size()) / 2;
    os << "===" << std::string(73, '-') << "===\n";
    os << std::string(padding, ' ') << header << "\n";
    os << "===" << std::string(73, '-') << "===\n";

    os << "  Total Execution Time: " << std::fixed << std::setprecision(3)
       << pipeline_timers_[op].GetTimePerSecond() << " seconds\n\n";
    os << "  ----Wall Time----  ----Name----\n";

    auto& map = pass_times_[op];
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
};
}  // namespace

void PassManager::EnableTiming() {
  AddInstrumentation(std::make_unique<PassTiming>());
}

}  // namespace infra
