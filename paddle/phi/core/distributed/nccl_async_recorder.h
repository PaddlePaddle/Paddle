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

class NCCLAsyncRecorder {
 public:
  NCCLAsyncRecorder(const phi::Place& place,
                    int rank,
                    int gid,
                    gpuStream_t stream,
                    CommType comm_type);
  ~NCCLAsyncRecorder() = default;

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

 private:
  phi::Place place_;
  int rank_;
  int gid_;
  gpuStream_t nccl_stream_;
  CommType comm_type_;

  gpuEvent_t nccl_start_event_;
  gpuEvent_t nccl_end_event_;

  bool is_start_;

  // std::chrono::high_resolution_clock::time_point start_time_;

 private:
  DISABLE_COPY_AND_ASSIGN(NCCLAsyncRecorder);
};

}  // namespace distributed
}  // namespace phi
