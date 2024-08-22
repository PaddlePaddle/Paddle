//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/heter_ccl_context.h"

#include <chrono>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

imperative::ParallelStrategy GetStrategy(int local_rank) {
  std::vector<std::string> eps = {"127.0.0.1:37580", "127.0.0.1:37581"};
  imperative::ParallelStrategy strategy;
  strategy.trainer_endpoints_ = eps;
  strategy.current_endpoint_ = eps[local_rank];
  strategy.nranks_ = eps.size();
  strategy.local_rank_ = local_rank;
  return strategy;
}

#ifdef PADDLE_WITH_NCCL
void AllReduceByStream(int local_rank, int device_id) {
  int data_size = 32;
  const auto& place = phi::GPUPlace(device_id);
  phi::GPUContext ctx(place);

  // heter_parallel_ctx
  imperative::HeterParallelContext hpc(GetStrategy(local_rank), device_id);

  // init
  hpc.Init();

  // input and output data
  framework::Variable* src_dev_var(new framework::Variable());
  auto* src_dev_tensor = src_dev_var->GetMutable<phi::DenseTensor>();
  src_dev_tensor->mutable_data<float>(common::make_ddim({data_size}), place);

  std::vector<float> src_vec;
  for (int i = 0; i < data_size; i++) {
    src_vec.push_back(1.0 + local_rank);
  }
  framework::TensorFromVector(src_vec, ctx, src_dev_tensor);
  ctx.Wait();

  framework::Variable* dst_dev_var(new framework::Variable());
  auto* dst_dev_tensor = dst_dev_var->GetMutable<phi::DenseTensor>();
  dst_dev_tensor->mutable_data<float>(common::make_ddim({data_size}), place);

  // call allreduce
  hpc.AllReduceByStream(*src_dev_var, dst_dev_var, 0, false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // check result
  std::vector<float> dst_vec;
  framework::TensorToVector(*dst_dev_tensor, ctx, &dst_vec);
  ctx.Wait();

  EXPECT_EQ(dst_vec.size(), src_vec.size());
  for (int i = 0; i < data_size; i++) {
    EXPECT_EQ(dst_vec[i], 3.0);
  }
}

TEST(AllReduceByStream, Run) {
  if (platform::GetGPUDeviceCount() >= 2) {
    std::thread t0(AllReduceByStream, 0, 0);
    std::thread t1(AllReduceByStream, 1, 1);
    t0.join();
    t1.join();
  }
}
#endif
