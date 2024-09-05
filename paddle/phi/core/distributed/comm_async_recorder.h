// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <string>

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
namespace distributed {

class CommAsyncRecorder {
 public:
  CommAsyncRecorder(const phi::Place& place, int gid, gpuStream_t stream);
  ~CommAsyncRecorder() = default;

  float GetTime() const;
  void StartRecord();
  void EndRecord();
  bool QueryEnd() const;
  bool QueryStart() const;
  bool IsStart() const { return is_start_; }
  void Start();

  int GetGid() const { return gid_; }

  float RecordTime() const;
  bool EventQuery(gpuEvent_t event) const;
  void EventDestroy();

  static void SynchronizeAllRecorders();

 private:
  phi::Place place_;
  int gid_;
  gpuStream_t nccl_stream_;

  gpuEvent_t nccl_start_event_;
  gpuEvent_t nccl_end_event_;

  bool is_start_;

 private:
  DISABLE_COPY_AND_ASSIGN(CommAsyncRecorder);
};

}  // namespace distributed
}  // namespace phi
