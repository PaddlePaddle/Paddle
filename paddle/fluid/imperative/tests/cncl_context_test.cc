/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thread>  // NOLINT

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/cncl_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

#include "gtest/gtest.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

// Node1: FLAGS_selected_mlus=0 PADDLE_TRAINER_ID=0 ./cncl_context_test
// Node2: FLAGS_selected_mlus=1 PADDLE_TRAINER_ID=1 ./cncl_context_test

int nrings = 1;
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

#if defined(PADDLE_WITH_CNCL)
void Broadcast(int local_rank, int device_id) {
  int data_size = 4;
  float test_data = 7;
  const auto& place = platform::MLUPlace(device_id);
  platform::MLUDeviceContext ctx(place);

  imperative::CNCLParallelContext cpc(GetStrategy(local_rank), place);

  // init
  cpc.Init();

  framework::Variable* src_dev_var(new framework::Variable());
  auto* src_dev_tensor = src_dev_var->GetMutable<framework::LoDTensor>();
  src_dev_tensor->mutable_data<float>(phi::make_ddim({data_size}), place);

  // fill data for rank 0 only
  std::vector<float> src_vec;
  if (local_rank == 0) {
    for (int i = 0; i < data_size; ++i) {
      src_vec.push_back(test_data);
    }
    framework::TensorFromVector(src_vec, ctx, src_dev_tensor);
  }
  ctx.Wait();

  // call broadcast
  cpc.Broadcast(src_dev_var, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // check result
  std::vector<float> dst_vec;
  framework::TensorToVector(*src_dev_tensor, ctx, &dst_vec);
  ctx.Wait();

  for (int i = 0; i < data_size; ++i) {
    EXPECT_EQ(dst_vec[i], test_data);
  }
}

TEST(Broadcast, Run) {
  if (platform::GetMLUDeviceCount() >= 2) {
    int local_rank = atoi(getenv("PADDLE_TRAINER_ID"));
    int device_id = atoi(getenv("FLAGS_selected_mlus"));
    Broadcast(local_rank, device_id);
  }
}

void AllReduceByStream(int local_rank, int device_id) {
  int data_size = 32;
  const auto& place = platform::MLUPlace(device_id);
  platform::MLUDeviceContext ctx(place);

  imperative::CNCLParallelContext cpc(GetStrategy(local_rank), place);

  // init
  cpc.Init();

  // input data
  framework::Variable* src_dev_var(new framework::Variable());
  auto* src_dev_tensor = src_dev_var->GetMutable<framework::LoDTensor>();
  src_dev_tensor->mutable_data<float>(phi::make_ddim({data_size}), place);

  // fill input data
  std::vector<float> src_vec;
  for (int i = 0; i < data_size; ++i) {
    src_vec.push_back(1.0 + local_rank);
  }
  framework::TensorFromVector(src_vec, ctx, src_dev_tensor);
  ctx.Wait();

  // output data
  framework::Variable* dst_dev_var(new framework::Variable());
  auto* dst_dev_tensor = dst_dev_var->GetMutable<framework::LoDTensor>();
  dst_dev_tensor->mutable_data<float>(phi::make_ddim({data_size}), place);

  // call allreduce
  cpc.AllReduceByStream(*src_dev_var, dst_dev_var, 0, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // check result
  std::vector<float> dst_vec;
  framework::TensorToVector(*dst_dev_tensor, ctx, &dst_vec);
  ctx.Wait();

  EXPECT_EQ(dst_vec.size(), src_vec.size());
  for (int i = 0; i < data_size; ++i) {
    EXPECT_EQ(dst_vec[i], 3.0);
  }
}

TEST(AllReduceByStream, Run) {
  if (platform::GetMLUDeviceCount() >= 2) {
    int local_rank = atoi(getenv("PADDLE_TRAINER_ID"));
    int device_id = atoi(getenv("FLAGS_selected_mlus"));
    AllReduceByStream(local_rank, device_id);
  }
}
#endif
