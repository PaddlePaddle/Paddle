/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <unistd.h>

#include <atomic>
#include "paddle/utils/Logging.h"

namespace paddle {

/*
 * if sparse_remote_updater is used, different ParameterServer could
 * be assigned with unbalanced gradients. the parameter value from
 * ParameterServer also be not balanced. the distribution of different
 * dimensions of sparse ids determines the unbalanced degree of data
 * distributed among all ParameterServers. Even distribution will
 * benifits cluster efficiency.
 * do check the unbalanced degree of gradients at runtime, crash program
 * if unbalanced distribution exhibts by default.
 */
class SparseParameterDistribution {
public:
  /// serviceNum means the number of ParameterServers
  explicit SparseParameterDistribution(size_t serviceNum);
  ~SparseParameterDistribution() {}
  /// collect data
  void probeDistribution(int serverId, size_t data);
  void checkAndResetDistribution();

private:
  std::vector<size_t> data_;
  std::atomic<size_t> totBytes_;

  /// after some batches, stop to check
  int batchPassed_;

  /// stat on unbalanced distribution found
  int unbalanceCnt_;
};
}  // namespace paddle
