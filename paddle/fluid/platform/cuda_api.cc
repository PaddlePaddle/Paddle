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

#include "paddle/fluid/platform/cuda_api.h"
#include "cuda_api.h"

namespace paddle {
namespace platform {

int CudaAPI::GetDeviceCount() {
  int count;
  PADDLE_ENFORCE(cudaGetDeviceCount(&count));
  return count;
}

void CudaAPI::SetDevice(int device) { PADDLE_ENFORCE(cudaSetDevice(device)); }
cudaEvent_t CudaAPI::CreateEvent(bool flag) {
  event_t event;
  if (flag) {
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&event, cudaEventDefault));
  } else {
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  return event;
}
cudaStream_t CudaAPI::CreateStream() {
  stream_t stream;
  PADDLE_ENFORCE(cudaStreamCreate(&stream));
  return stream;
}
cudaStream_t CudaAPI::CreateStreamWithFlag(int flag) {
  stream_t stream;
  PADDLE_ENFORCE(cudaStreamCreateWithFlags(&stream, flag));
  return stream;
}
void CudaAPI::DestroyStream(stream_t *stream) {
  PADDLE_ENFORCE(cudaStreamDestroy(*stream));
}
void CudaAPI::RecordEvent(event_t event) {
  PADDLE_ENFORCE(cudaEventRecord(event));
}
bool CudaAPI::QueryEvent(event_t event) {
  return cudaEventQuery(event) == cudaSuccess;
}
void CudaAPI::SyncEvent(event_t event) {
  PADDLE_ENFORCE(cudaEventSynchronize(event));
}
void CudaAPI::SyncStream(stream_t stream) {
  PADDLE_ENFORCE(cudaStreamSynchronize(stream));
}

}  // namespace platform
}  // namespace paddle
