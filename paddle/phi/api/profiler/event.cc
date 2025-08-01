// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/profiler/event.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "glog/logging.h"
#endif

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

CudaEvent::CudaEvent() {
#ifdef PADDLE_WITH_HIP
  hipEventCreateWithFlags(&event_, flags_);
#else
  cudaEventCreateWithFlags(&event_, flags_);
#endif
  VLOG(4) << "CudaEvent " << event_;
}

CudaEvent::CudaEvent(unsigned int flags) : flags_(flags) {
#ifdef PADDLE_WITH_HIP
  hipEventCreateWithFlags(&event_, flags_);
#else
  cudaEventCreateWithFlags(&event_, flags_);
#endif
  VLOG(4) << "CudaEvent " << event_;
}

bool CudaEvent::Query() {
#ifdef PADDLE_WITH_HIP
  gpuError_t err = hipEventQuery(event_);
  if (err == hipSuccess) {
    return true;
  }
  if (err == hipErrorNotReady) {
    return false;
  }
#else
  gpuError_t err = cudaEventQuery(event_);
  if (err == cudaSuccess) {
    return true;
  }
  if (err == cudaErrorNotReady) {
    return false;
  }
#endif
  PADDLE_ENFORCE_GPU_SUCCESS(err);
  return false;
}

float CudaEvent::ElapsedTime(CudaEvent *end_event) {
  float milliseconds = 0;
#ifdef PADDLE_WITH_HIP
  hipEventSynchronize(end_event->GetRawCudaEvent());
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipEventElapsedTime(&milliseconds, event_, end_event->GetRawCudaEvent()));
#else
  cudaEventSynchronize(end_event->GetRawCudaEvent());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(
      &milliseconds, event_, end_event->GetRawCudaEvent()));
#endif
  return milliseconds;
}

#endif

}  // namespace phi
