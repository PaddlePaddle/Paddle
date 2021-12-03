// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/kernel/test_kernels.h"

#include <llvm/ADT/FunctionExtras.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"
#include "paddle/infrt/tensor/dense_host_tensor.h"

using infrt::host_context::Attribute;
using infrt::host_context::MlirFunctionExecutable;
using infrt::host_context::RemainingArguments;

namespace infrt::kernel {
namespace {
class BenchmarkStats {
 public:
  BenchmarkStats(std::string name,
                 int num_warmup_runs,
                 int max_count,
                 std::chrono::microseconds benchmark_duration)
      : name_{name},
        num_warmup_runs_{num_warmup_runs},
        max_count_{max_count},
        benchmark_duration_{benchmark_duration} {}

  void StartRun() {
    ++cur_count_;
    // Start recording CPU time.
    cur_start_walltime_ = std::chrono::steady_clock::now();
    cur_start_cpu_ = std::clock();
  }

  void StopRun() {
    // Do not collect the runtime statistics if we are still in the warm up
    // period.
    if (cur_count_ <= num_warmup_runs_) return;

    // Stop the CPU timer.
    std::clock_t cur_stop_cpu_ = std::clock();

    // Stop the wall clock timer.
    auto cur_stop_walltime_ = std::chrono::steady_clock::now();

    // Collect the wall clock duration.
    auto duration_walltime_ = cur_stop_walltime_ - cur_start_walltime_;
    run_times_walltime_.push_back(duration_walltime_);

    // Collect the CPU duration in microseconds.
    // First cast to integer that represents microseconds with truncation, as
    // does std::chrono::duration_cast. Then cast to std::chrono::microseconds.
    std::clock_t duration_cpu_raw = cur_stop_cpu_ - cur_start_cpu_;
    auto duration_cpu_ = static_cast<std::chrono::nanoseconds>(
        static_cast<int64_t>(1e9 * duration_cpu_raw / CLOCKS_PER_SEC));

    run_times_cpu_.push_back(duration_cpu_);

    total_duration_walltime_ += duration_walltime_;
    total_duration_cpu_ += duration_cpu_;
  }
  // Return if we should we run more rounds.
  bool MoreRun() const {
    return cur_count_ < max_count_ + num_warmup_runs_ &&
           total_duration_walltime_ < benchmark_duration_;
  }

  // Summarize the benchmark results.
  void Summarize() {
    std::sort(run_times_walltime_.begin(), run_times_walltime_.end());
    std::sort(run_times_cpu_.begin(), run_times_cpu_.end());

    auto percentile = [](
        double p, const std::vector<std::chrono::nanoseconds> &run_times) {
      assert(p >= 0.0 && p <= 1.0);
      return run_times[run_times.size() * p];
    };

    // BM: prefix is added to make grepping results from lit output easier.
    std::string prefix;
    llvm::raw_string_ostream(prefix) << "BM:" << name_ << ':';
    auto cpu_utilization =
        total_duration_cpu_.count() * 100.0 / total_duration_walltime_.count();

    llvm::outs() << prefix << "Count: " << run_times_walltime_.size() << '\n';
    llvm::outs() << prefix
                 << "Duration(ns): " << total_duration_walltime_.count()
                 << '\n';
    llvm::outs() << prefix
                 << "Time Min(ns): " << run_times_walltime_.front().count()
                 << '\n';
    llvm::outs() << prefix
                 << "Time Max(ns): " << run_times_walltime_.back().count()
                 << '\n';
    llvm::outs() << prefix << "Time 50%(ns): "
                 << percentile(0.5, run_times_walltime_).count() << '\n';
    llvm::outs() << prefix << "Time 95%(ns): "
                 << percentile(0.95, run_times_walltime_).count() << '\n';
    llvm::outs() << prefix << "Time 99%(ns): "
                 << percentile(0.99, run_times_walltime_).count() << '\n';
    // Log CPU time statistics.
    llvm::outs() << prefix
                 << "CPU Duration(ns): " << total_duration_cpu_.count() << '\n';
    llvm::outs() << prefix << "CPU Min(ns): " << run_times_cpu_.front().count()
                 << '\n';
    llvm::outs() << prefix << "CPU Max(ns): " << run_times_cpu_.back().count()
                 << '\n';
    llvm::outs() << prefix
                 << "CPU 50%(ns): " << percentile(0.5, run_times_cpu_).count()
                 << '\n';
    llvm::outs() << prefix
                 << "CPU 95%(ns): " << percentile(0.95, run_times_cpu_).count()
                 << '\n';
    llvm::outs() << prefix
                 << "CPU 99%(ns): " << percentile(0.99, run_times_cpu_).count()
                 << '\n';
    llvm::outs() << prefix << "CPU utilization(percent): " << cpu_utilization
                 << "\n";
    llvm::outs().flush();
  }

 private:
  const std::string name_;
  const int num_warmup_runs_;
  const int max_count_;
  int cur_count_ = 0;
  const std::chrono::nanoseconds benchmark_duration_;
  std::chrono::nanoseconds total_duration_walltime_{};
  std::chrono::nanoseconds total_duration_cpu_{};
  std::chrono::time_point<std::chrono::steady_clock> cur_start_walltime_{};
  std::clock_t cur_start_cpu_;
  std::vector<std::chrono::nanoseconds> run_times_walltime_;
  // CPU run times in microseconds.
  std::vector<std::chrono::nanoseconds> run_times_cpu_;
};

}  // anonymous namespace

// This op benchmarks the input function by running the function in a loop
// up to a max count or max time as specified in the function's attributes.
//
// Attributes:
// duration_secs: Benchmark duration in seconds.
// max_count: Max run count of input function.
// name: The name used to tag the benchmark results.
// num_warmup_runs: Number of warm up runs before benchmarking starts.
// fn: The input function to be benchmarked.
static void benchmark(RemainingArguments args,
                      host_context::RemainingResults results,
                      Attribute<int32_t> duration_secs,
                      Attribute<int32_t> max_count,
                      Attribute<std::string> name,
                      Attribute<int32_t> num_warmup_runs,
                      Attribute<MlirFunctionExecutable *> fn) {
  BenchmarkStats bm_stats{name.get(),
                          num_warmup_runs.get(),
                          max_count.get(),
                          std::chrono::seconds(duration_secs.get())};

  while (bm_stats.MoreRun()) {
    bm_stats.StartRun();
    fn.get()->Execute(args.values(), results.values(), true);
    bm_stats.StopRun();
  }
  bm_stats.Summarize();
}

// Just copy the input to the result.
tensor::DenseHostTensor ShadowCopyTensor(tensor::DenseHostTensor src) {
  return src;
}

void RegisterTestKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("infrt.benchmark", INFRT_KERNEL(benchmark));
  registry->AddKernel("infrt.test.shadow_copy_tensor",
                      INFRT_KERNEL(ShadowCopyTensor));
}

}  // namespace infrt::kernel
