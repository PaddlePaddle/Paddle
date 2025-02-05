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

#include "paddle/cinn/utils/multi_threading.h"

#include <glog/logging.h>

#include <future>
#include <thread>
#include <utility>
#include <vector>
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

SequenceDispatcher::SequenceDispatcher(int begin, int end, int step)
    : end_(end), step_(step), index_(begin) {
  PADDLE_ENFORCE_LE(
      begin,
      end,
      ::common::errors::InvalidArgument("begin[%d] > end[%d]", begin, end));
  PADDLE_ENFORCE_GT(
      step, 0, ::common::errors::InvalidArgument("step is less than 0."));
}

int SequenceDispatcher::Next() const {
  int idx = -1;
  if (index_ >= end_ || (idx = index_.fetch_add(step_)) >= end_) {
    return -1;
  }

  return idx;
}

void parallel_run(const WorkerFuncType& fn,
                  JobDispatcher&& dispatcher,
                  int num_threads) {
  if (num_threads == -1 || num_threads > std::thread::hardware_concurrency()) {
    num_threads = std::thread::hardware_concurrency();
  }
  PADDLE_ENFORCE_GT(num_threads,
                    0,
                    ::common::errors::PreconditionNotMet(
                        "num_threads should be greater than 0"));

  // worker function of a thread
  auto worker = [&fn, &dispatcher](int tid) -> int {
    int index = -1, counter = 0;
    while ((index = dispatcher.Next()) != -1) {
      VLOG(5) << "Thread-" << tid << " process at index: " << index;
      fn(index);
      ++counter;
    }
    return counter;
  };

  std::vector<std::future<int>> futures;
  std::vector<std::thread> threads;
  // The first thread runs inplace, and other `num_threads - 1` threads launched
  // with std::future to run asynchronously
  if (num_threads > 1) {
    futures.reserve(num_threads - 1);
    threads.reserve(num_threads - 1);
    for (int tid = 1; tid < num_threads; ++tid) {
      std::packaged_task<int(int)> task(worker);
      futures.emplace_back(task.get_future());
      threads.emplace_back(std::move(task), tid);
    }
  }

  // wait results and catch exceptions
  try {
    int tid = 0;
    int counter = worker(tid);
    VLOG(4) << "Thread-0  process " << counter << " tasks.";

    for (auto& future : futures) {
      counter = future.get();
      ++tid;
      VLOG(4) << "Thread-" << tid << " process " << counter << " tasks.";
    }

    // join threads
    for (auto&& thread : threads) {
      thread.join();
    }
  } catch (::common::EnforceNotMet& ex) {
    LOG(ERROR) << ex.error_str();
    PADDLE_THROW(
        ::common::errors::Fatal("Parallel compile Paddle enforce error"));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Parallel compile error " << e.what();
    PADDLE_THROW(::common::errors::Fatal("Parallel compile std::exception"));
  } catch (...) {
    PADDLE_THROW(::common::errors::Fatal("Parallel compile unknow exception"));
  }
}

}  // namespace utils
}  // namespace cinn
