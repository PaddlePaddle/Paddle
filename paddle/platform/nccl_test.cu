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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/dynload/nccl.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/gpu_info.h"

#include <thrust/device_vector.h>
#include <memory>
#include <vector>

static int dev_count = 0;

namespace paddle {
namespace platform {

TEST(NCCL, init) {
  std::vector<ncclComm_t> comms;
  comms.resize(dev_count);
  PADDLE_ENFORCE(dynload::ncclCommInitAll(comms.data(), dev_count, nullptr));
  for (int i = 0; i < dev_count; ++i) {
    dynload::ncclCommDestroy(comms[i]);
  }
}

template <typename T>
struct PerThreadData {
  thrust::device_vector<T> send_buff;
  thrust::device_vector<T> recv_buff;
  CUDADeviceContext dev_ctx;

  T* SendBuff() { return thrust::raw_pointer_cast(send_buff.data()); }

  T* RecvBuff() { return thrust::raw_pointer_cast(recv_buff.data()); }

  PerThreadData(int gpu_id, size_t size) : dev_ctx(GPUPlace(gpu_id)) {
    send_buff.resize(size);
    for (size_t i = 0; i < size; ++i) {
      send_buff[i] = static_cast<T>(i);
    }
    recv_buff.resize(size);
  }
};

static constexpr int ELEM_COUNT = 10000;

TEST(NCCL, all_reduce) {
  std::vector<ncclComm_t> comms;
  comms.resize(dev_count);
  VLOG(1) << "Initializing ncclComm";
  PADDLE_ENFORCE(dynload::ncclCommInitAll(comms.data(), dev_count, nullptr));
  VLOG(1) << "ncclComm initialized";
  VLOG(1) << "Creating thread data";
  std::vector<std::unique_ptr<PerThreadData<double>>> data;
  data.reserve(dev_count);
  for (int i = 0; i < dev_count; ++i) {
    VLOG(1) << "Creating thread data for device " << i;
    SetDeviceId(i);
    data.emplace_back(new PerThreadData<double>(i, ELEM_COUNT));
  }
  VLOG(1) << "Thread data created";

  VLOG(1) << "Check send_buf data";
  for (int i = 0; i < dev_count; ++i) {
    VLOG(1) << "Check on device " << i;
    SetDeviceId(i);
    thrust::host_vector<double> tmp = data[i]->send_buff;
    for (size_t j = 0; j < tmp.size(); ++j) {
      ASSERT_NEAR(static_cast<double>(j), tmp[j], 1e-5);
    }
  }

  VLOG(1) << "Invoking ncclAllReduce";

  for (int i = 0; i < dev_count; ++i) {
    VLOG(1) << "Invoking ncclAllReduce with device " << i;
    SetDeviceId(i);
    PADDLE_ENFORCE(dynload::ncclAllReduce(
        data[i]->SendBuff(), data[i]->RecvBuff(), ELEM_COUNT, ncclDouble,
        ncclSum, comms[i], data[i]->dev_ctx.stream()));
    VLOG(1) << "Invoked ncclAllReduce for device " << i;
  }

  VLOG(1) << "Invoked ncclAllReduce";

  VLOG(1) << "Sync devices";
  for (int i = 0; i < dev_count; ++i) {
    VLOG(1) << "Sync device " << i;
    SetDeviceId(i);
    data[i]->dev_ctx.Wait();
  }
  VLOG(1) << "device synced";

  for (int i = 0; i < dev_count; ++i) {
    SetDeviceId(i);
    VLOG(1) << "Checking vector on device " << i;
    thrust::host_vector<double> tmp = data[i]->recv_buff;
    for (size_t j = 0; j < tmp.size(); ++j) {
      auto elem = static_cast<double>(j);
      elem *= dev_count;
      ASSERT_NEAR(tmp[j], elem, 1e-4);
    }
  }

  for (int i = 0; i < dev_count; ++i) {
    dynload::ncclCommDestroy(comms[i]);
  }
}
}  // namespace platform
}  // namespace paddle

int main(int argc, char** argv) {
  dev_count = paddle::platform::GetCUDADeviceCount();
  if (dev_count <= 1) {
    LOG(WARNING)
        << "Cannot test multi-gpu nccl, because the CUDA device count is "
        << dev_count;
    return 0;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
