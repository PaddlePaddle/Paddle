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

#include "paddle/fluid/imperative/nccl_context.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;

imperative::ParallelStrategy GetStrategy(int local_rank) {
  std::vector<std::string> eps = {"127.0.0.1:9866", "127.0.0.1:9867"};
  imperative::ParallelStrategy strategy;
  strategy.trainer_endpoints_ = eps;
  strategy.current_endpoint_ = eps[local_rank];
  strategy.nranks_ = 2;
  strategy.local_rank_ = local_rank;
  return strategy;
}

#ifdef PADDLE_WITH_CUDA
void BcastNCCLId(int local_rank, ncclUniqueId *nccl_id) {
  auto strategy = GetStrategy(local_rank);
  platform::CUDAPlace gpu(local_rank);
  imperative::NCCLParallelContext ctx(strategy, gpu);
  ctx.BcastNCCLId(nccl_id, 0);
}

TEST(BcastNCCLId, Run) {
  ncclUniqueId nccl_id;
  platform::dynload::ncclGetUniqueId(&nccl_id);
  std::thread t(BcastNCCLId, 0, &nccl_id);

  ncclUniqueId recv_nccl_id;
  BcastNCCLId(1, &recv_nccl_id);

  t.join();
  EXPECT_EQ(0, std::memcmp(nccl_id.internal, recv_nccl_id.internal,
                           NCCL_UNIQUE_ID_BYTES));
}
#endif
