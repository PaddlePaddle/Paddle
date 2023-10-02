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

#include "paddle/cinn/auto_schedule/measure/schedule_measurer.h"

#include <exception>

#include "paddle/cinn/utils/multi_threading.h"

namespace cinn {
namespace auto_schedule {

ScheduleMeasurer::ScheduleMeasurer(ScheduleBuilder* builder,
                                   ScheduleRunner* runner,
                                   int num_threads)
    : builder_(builder), runner_(runner), num_threads_(num_threads) {}

std::vector<MeasureResult> ScheduleMeasurer::Measure(
    const std::vector<MeasureInput>& inputs) {
  if (inputs.empty()) {
    LOG(WARNING) << "inputs is empty";
    return {};
  }
  std::vector<BuildResult> build_results(inputs.size());
  std::vector<MeasureResult> results(inputs.size());

  // define how to build a candidate with the specified index
  auto build_fn =
      [builder = builder_, &inputs, &build_results, &results](int index) {
        VLOG(6) << "Build candidate index: " << index;
        auto m_start = std::chrono::steady_clock::now();
        try {
          build_results[index] = builder->Build(inputs[index]);
        } catch (std::exception& e) {
          results[index].error_msg =
              utils::StringFormat("Build failed, error: %s\n", e.what());
        }
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - m_start);
        results[index].elapsed_time += static_cast<double>(time_span.count());
      };

  // define how to run a candidate with the specified index
  auto run_fn =
      [runner = runner_, &inputs, &build_results, &results](int index) {
        VLOG(6) << "Run candidate index: " << index;
        auto m_start = std::chrono::steady_clock::now();
        try {
          // if error occurred in building, then skip running
          if (results[index].error_msg.empty()) {
            results[index] = runner->Run(inputs[index], build_results[index]);
          }
        } catch (std::exception& e) {
          results[index].error_msg =
              utils::StringFormat("Run failed, error: %s\n", e.what());
        }
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - m_start);
        results[index].elapsed_time += static_cast<double>(time_span.count());
      };

  // measure a candidate by calling build and run successively
  auto measure_fn = [&build_fn, &run_fn](int index) {
    build_fn(index);
    run_fn(index);
  };
  // default num_threads_ is 1 and in that case it will perform all measurements
  // sequentially inplace.
  utils::parallel_run(
      measure_fn, utils::SequenceDispatcher(0, inputs.size()), num_threads_);

  VLOG(4) << "Measure " << inputs.size() << " candidates";
  return results;
}

}  // namespace auto_schedule
}  // namespace cinn
