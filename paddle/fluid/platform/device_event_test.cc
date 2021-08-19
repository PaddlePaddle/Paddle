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

#include "paddle/fluid/platform/device_event.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using ::paddle::platform::kCUDA;
using ::paddle::platform::kCPU;
USE_EVENT(kCUDA);
USE_EVENT_WAIT(kCUDA, kCUDA)
USE_EVENT_WAIT(kCPU, kCUDA)

TEST(DeviceEvent, CUDA) {
  VLOG(1) << "In Test";
  using paddle::platform::CUDAPlace;
  using paddle::platform::DeviceOption;
  using paddle::platform::DeviceEvent;
  using paddle::platform::DeviceContextPool;
  using paddle::platform::DeviceType;

  auto& pool = DeviceContextPool::Instance();
  auto place = CUDAPlace(0);
  auto* context =
      static_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
  int device_type = static_cast<int>(DeviceType::CUDA);
  DeviceOption dev_opt(device_type, place.device);

  ASSERT_NE(context, nullptr);
  // case 1. test for event_creator
  DeviceEvent event(dev_opt);
  ASSERT_NE(event.GetEvent().get(), nullptr);
  // case 2. test for event_recorder
  event.Record(place, context);
  bool status = event.Query();
  ASSERT_EQ(status, false);
  // case 3. test for event_finisher
  event.Finish();
  status = event.Query();
  ASSERT_EQ(status, true);

  // case 4. test for event_waiter
  float *src_fp32, *dst_fp32;
  int size = 1000000 * sizeof(float);
  cudaMallocHost(reinterpret_cast<void**>(&src_fp32), size);
  cudaMalloc(reinterpret_cast<void**>(&dst_fp32), size);
  cudaMemcpyAsync(dst_fp32, src_fp32, size, cudaMemcpyHostToDevice,
                  context->stream());
  event.Record(place, context);  // step 1. record it
  status = event.Query();
  ASSERT_EQ(status, false);

  event.Wait(kCUDA, context);  // step 2. add streamWaitEvent
  status = event.Query();
  ASSERT_EQ(status, false);  // async

  event.Wait(kCPU, context);  // step 3. EventSynchornize
  status = event.Query();
  ASSERT_EQ(status, true);  // sync

  // release resource
  cudaFree(dst_fp32);
  cudaFreeHost(src_fp32);
}
#endif
