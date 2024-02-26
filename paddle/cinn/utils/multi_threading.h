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
#include <atomic>
#include <functional>

namespace cinn {
namespace utils {

// function prototype that takes a index of job as argument and complete the
// specified job
using WorkerFuncType = std::function<void(int index)>;

// This class defines which job will be executed in the next turn,
// and returns the next job id through `Next` function, which will be used in
// multi-threads context It should be used with a function instance of
// WorkerFuncType which takes the index as argument.
class JobDispatcher {
 public:
  // Attention!! this interface must be implemented to be thread-safe
  virtual int Next() const = 0;
};

// This dispatcher simulates the execution of a for loop,
// that is traversing from `begin`(inclusive) to `end`(exclusive)
// with striding over `step` a time.
class SequenceDispatcher : public JobDispatcher {
 public:
  SequenceDispatcher(int begin, int end, int step = 1);

  int Next() const override;

 private:
  // the maximum index of extent
  int end_;
  // the traversal step to the next one
  int step_;
  // current index, using atomic to ensure thread-safe
  mutable std::atomic<int> index_;
};

/**
 * \brief A general function to run a batch of jobs in parallel
 * \param fn A instance of WorkerFuncType, which defines how to complete a
 * specified job \param dispatcher A instance of JobDispatcher, which pops index
 * of the next job \param num_threads The number of threads used to run jobs, -1
 * means utilizing the maximum limit of hardware
 */
void parallel_run(const WorkerFuncType& fn,
                  JobDispatcher&& dispatcher,
                  int num_threads = -1);

}  // namespace utils
}  // namespace cinn
