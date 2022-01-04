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

#include <thread>  // NOLINT

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

#include "gtest/gtest.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
void BcastNCCLId(int local_rank, std::vector<ncclUniqueId>* nccl_ids) {
  auto strategy = GetStrategy(local_rank);
  int server_fd = platform::CreateListenSocket(strategy.current_endpoint_);

  platform::CUDAPlace gpu(local_rank);
  imperative::NCCLParallelContext ctx(strategy, gpu);
  ctx.BcastNCCLId(*nccl_ids, 0, server_fd);

  platform::CloseSocket(server_fd);
}

TEST(BcastNCCLId, Run) {
  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    platform::dynload::ncclGetUniqueId(&nccl_ids[i]);
  }

  std::thread t(BcastNCCLId, 0, &nccl_ids);

  std::vector<ncclUniqueId> recv_nccl_ids;
  recv_nccl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    platform::dynload::ncclGetUniqueId(&recv_nccl_ids[i]);
  }
  BcastNCCLId(1, &recv_nccl_ids);

  t.join();
  for (int i = 0; i < nrings; ++i) {
    EXPECT_EQ(0, std::memcmp(nccl_ids[i].internal, recv_nccl_ids[i].internal,
                             NCCL_UNIQUE_ID_BYTES));
  }
}

void Broadcast(int local_rank, int device_id) {
  int data_size = 4;
  float test_data = 7;
  const auto& place = platform::CUDAPlace(device_id);
  platform::CUDADeviceContext ctx(place);

  imperative::NCCLParallelContext npc(GetStrategy(local_rank), place);

  // init
  npc.Init();

  framework::Variable* src_dev_var(new framework::Variable());
  auto* src_dev_tensor = src_dev_var->GetMutable<framework::LoDTensor>();
  src_dev_tensor->mutable_data<float>(framework::make_ddim({data_size}), place);

  // fill data for rank 0 only
  std::vector<float> src_vec;
  if (local_rank == 0) {
    for (int i = 0; i < data_size; i++) {
      src_vec.push_back(test_data);
    }
    framework::TensorFromVector(src_vec, ctx, src_dev_tensor);
  }
  ctx.Wait();

  npc.Broadcast(src_dev_var, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // check result
  std::vector<float> dst_vec;
  framework::TensorToVector(*src_dev_tensor, ctx, &dst_vec);
  ctx.Wait();

  for (int i = 0; i < data_size; i++) {
    EXPECT_EQ(dst_vec[i], test_data);
  }
}

TEST(Broadcast, Run) {
  if (platform::GetGPUDeviceCount() >= 2) {
    std::thread t0(Broadcast, 0, 0);
    std::thread t1(Broadcast, 1, 1);
    t0.join();
    t1.join();
  }
}
#endif
