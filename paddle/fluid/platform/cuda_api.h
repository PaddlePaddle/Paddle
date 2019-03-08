// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <cuda.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA

struct CudaAPI {
  using stream_t = cudaStream_t;
  using event_t = cudaEvent_t;

  static int GetDeviceCount();

  static void SetDevice(int device);

  static cudaEvent_t CreateEvent(bool flag);

  static cudaStream_t CreateStream();

  static cudaStream_t CreateStreamWithFlag(int flag);

  static void DestroyStream(stream_t *stream);

  static void RecordEvent(event_t event);

  static bool QueryEvent(event_t event);

  static void SyncEvent(event_t event);

  static void SyncStream(stream_t stream);
};

#endif  // PADDLE_WITH_CUDA

}  // namespace platform
}  // namespace paddle
