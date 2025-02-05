//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/bkcl_context.h"

#include <thread>  // NOLINT

#include "gtest/gtest.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;

int nrings = 2;
imperative::ParallelStrategy GetStrategy(int local_rank) {
  std::vector<std::string> eps = {"127.0.0.1:9866", "localhost:9867"};
  imperative::ParallelStrategy strategy;
  strategy.trainer_endpoints_ = eps;
  strategy.current_endpoint_ = eps[local_rank];
  strategy.nranks_ = 2;
  strategy.local_rank_ = local_rank;
  strategy.nrings_ = nrings;
  return strategy;
}

#if defined(PADDLE_WITH_XPU_BKCL)
void BcastBKCLId(int local_rank, std::vector<BKCLUniqueId>* bkcl_ids) {
  auto strategy = GetStrategy(local_rank);
  phi::XPUPlace xpu(local_rank);
  imperative::BKCLParallelContext ctx(strategy, xpu);
  ctx.BcastBKCLId(*bkcl_ids, 0);
}

TEST(BcastBKCLId, Run) {
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    bkcl_get_unique_id(&bkcl_ids[i]);
  }

  std::thread t(BcastBKCLId, 0, &bkcl_ids);

  std::vector<BKCLUniqueId> recv_bkcl_ids;
  recv_bkcl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    bkcl_get_unique_id(&recv_bkcl_ids[i]);
  }
  BcastBKCLId(1, &recv_bkcl_ids);

  t.join();
  for (int i = 0; i < nrings; ++i) {
    EXPECT_EQ(
        0, std::memcmp(&bkcl_ids[i], &recv_bkcl_ids[i], BKCL_UNIQUE_ID_BYTES));
  }
}
#endif
