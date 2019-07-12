// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/utils/benchmark.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace paddle::inference;  // NOLINT
TEST(Benchmark, basic) {
  Benchmark benchmark;
  benchmark.SetName("key0");
  benchmark.SetBatchSize(10);
  benchmark.SetUseGpu();
  benchmark.SetLatency(220);
  LOG(INFO) << "benchmark:\n" << benchmark.SerializeToString();
}

TEST(Benchmark, PersistToFile) {
  Benchmark benchmark;
  benchmark.SetName("key0");
  benchmark.SetBatchSize(10);
  benchmark.SetUseGpu();
  benchmark.SetLatency(220);

  benchmark.PersistToFile("1.log");
  benchmark.PersistToFile("2.log");
  benchmark.PersistToFile("3.log");
}
