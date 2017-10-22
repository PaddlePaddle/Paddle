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
#include "paddle/platform/dynload/nccl.h"
#include "paddle/platform/gpu_info.h"

#include <vector>

static int dev_count = 0;

namespace paddle {
namespace platform {

TEST(NCCL, init) {
  std::vector<ncclComm_t> comms;
  comms.resize(dev_count);

  auto status = dynload::ncclCommInitAll(comms, nGPUs);
  PADDLE_ENFORCE(status);
  for (int i = 0; i < dev_count; ++i) {
    PADDLE_ENFORCE(ncclCommDestroy(comms[i]));
  }
}
}
}

int main(int argc, char** argv) {
  dev_count = paddle::platform::GetCUDADeviceCount();
  if (dev_count <= 1) {
    LOG(WARN) << "Cannot test multi-gpu nccl, because the CUDA device count is "
              << dev_count;
    return 0;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}