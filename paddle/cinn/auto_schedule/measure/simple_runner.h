// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/measure/measure.h"
#include "paddle/cinn/hlir/framework/instruction.h"

namespace cinn {
namespace auto_schedule {

// This class utilize the built instructions to execute the generated
// kernels and count the elapsed time as the measurement of performance
class SimpleRunner : public ScheduleRunner {
 public:
  explicit SimpleRunner(int repeat_times);

  MeasureResult Run(const MeasureInput& input,
                    const BuildResult& build_result) override;

 private:
  std::map<std::string, cinn_pod_value_t> PrepareArgs(
      const MeasureInput& input,
      const BuildResult& build_result,
      hlir::framework::Scope* temp_scope);

 private:
  // The repeat times of running instructions,
  // this runner will return the average time
  const int repeat_times_;
};

}  // namespace auto_schedule
}  // namespace cinn
