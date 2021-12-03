/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_MLU
#include <vector>
#include "cnrt.h"
#include "cnnl.h"
#include "cn_api.h"

namespace paddle {

using cnStatus = CNresult;
using cnrtStatus = cnrtRet_t;
using cnnlStatus = cnnlStatus_t;
using mluStream = cnrtQueue_t;
using mluCnnlHandle = cnnlHandle_t;                                                            
using mluEventHandle = CNnotifier;
using mluDeviceHandle = CNdev;

typedef struct {
  int x;
  int y;
  int z;
} mluDim3;

namespace platform {

//! Get the driver version of the ith MLU.
mluDim3 GetMLUDriverVersion(int id);

//! Get the total number of MLU devices in system.
int GetMLUDeviceCount();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetMLUSelectedDevices();

//! Get the current MLU device id in system.
int GetMLUCurrentDeviceId();

//! Set the MLU device id for next execution.
void SetMLUDeviceId(int device_id);

//! Get a handle of device ids.
void GetMLUDeviceHandle(int device_ordinal, mluDeviceHandle* device);

//! Get the compute capability of the ith MLU (format: major * 10 + minor)
int GetMLUComputeCapability(int id);

class MLUDeviceGuard {
 public:
  explicit inline MLUDeviceGuard(int dev_id) {
    int prev_id = platform::GetMLUCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      platform::SetMLUDeviceId(dev_id);
    }
  }

  inline ~MLUDeviceGuard() {
    if (prev_id_ != -1) {
      platform::SetMLUDeviceId(prev_id_);
    }
  }

  MLUDeviceGuard(const MLUDeviceGuard& o) = delete;
  MLUDeviceGuard& operator=(const MLUDeviceGuard& o) = delete;

 private:
  int prev_id_{-1};
};

}  // namespace platform
}  // namespace paddle

#endif