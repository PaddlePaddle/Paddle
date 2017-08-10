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

#include <unistd.h>

#include "paddle/utils/Logging.h"

#include "paddle/utils/Flags.h"

#include "SparseParameterDistribution.h"

DEFINE_bool(check_sparse_distribution_in_pserver,
            false,
            "check whether sparse parameter exhibts balanced distribution at "
            "all pservers");
DEFINE_bool(show_check_sparse_distribution_log,
            false,
            "show logs details for sparse parameter distribution in pserver");
DEFINE_int32(check_sparse_distribution_batches,
             100,
             "run sparse parameter distribution check for N batches");
DEFINE_double(
    check_sparse_distribution_ratio,
    0.6,
    "if parameters dispatched to different pservers exhibit unbalanced "
    " distribution for check_sparse_distribution_ratio * "
    " check_sparse_distribution_batches times, crash program");
DEFINE_double(check_sparse_distribution_unbalance_degree,
              2.0,
              "the ratio of maximum data size and minimun data size for "
              "different pserver");

namespace paddle {

SparseParameterDistribution::SparseParameterDistribution(size_t serviceNum) {
  totBytes_ = 0;
  data_.resize(serviceNum);

  batchPassed_ = 0;
  unbalanceCnt_ = 0;
}

void SparseParameterDistribution::probeDistribution(int serverId,
                                                    size_t dataSize) {
  if (!FLAGS_check_sparse_distribution_in_pserver ||
      batchPassed_ > FLAGS_check_sparse_distribution_batches) {
    return;
  }

  CHECK_LT((size_t)serverId, data_.size())
      << "invalid sparse parameter distribution probe";

  data_[serverId] += dataSize;
  totBytes_ += dataSize;
}

void SparseParameterDistribution::checkAndResetDistribution() {
  if (!FLAGS_check_sparse_distribution_in_pserver ||
      batchPassed_ >= FLAGS_check_sparse_distribution_batches) {
    return;
  }

  /// at runtime, prepareSendData is called by many contexts,
  /// so need to check if data is avaiable.
  if (!totBytes_) {
    return;
  }

  /// check if distribution is balanced
  auto avgSize = totBytes_ / data_.size();
  auto unbalanceDegree = FLAGS_check_sparse_distribution_unbalance_degree;
  for (auto& dataSize : data_) {
    if (dataSize > unbalanceDegree * avgSize ||
        dataSize * unbalanceDegree < avgSize) {
      unbalanceCnt_++;
      break;
    }
  }

  auto printData = [&]() {
    std::stringstream ss;
    for (auto& dataSize : data_) {
      ss << dataSize * 0.001 << "KB ";
    }
    ss << std::endl;
    LOG(INFO) << ss.str();
  };

  /// show all sparse data size for different pserver
  if (FLAGS_show_check_sparse_distribution_log) {
    LOG(INFO) << "sparse distribution:";
    printData();
  }

  totBytes_ = 0;
  batchPassed_++;

  if (batchPassed_ == FLAGS_check_sparse_distribution_batches) {
    LOG(INFO) << "show last parameter distribution sample:";
    printData();
    LOG(INFO) << "total unbalanced batches: " << unbalanceCnt_
              << " in passed batches: " << batchPassed_;
    CHECK_LE((float)unbalanceCnt_ / (float)batchPassed_,
             FLAGS_check_sparse_distribution_ratio)
        << "unbalanced sparse parameter distribution for different pserver. "
        << "it could be caused by unbalanced sparse ids distribution, try "
        << "to shuffle dimensions in input samples";
  }

  std::fill(data_.begin(), data_.end(), 0);
}
}  // namespace paddle
