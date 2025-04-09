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
#include <vector>
#include "cuda_runtime.h"  // NOLINT
#include "paddle/common/enforce.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"

namespace phi {

class EventPair {
  DISABLE_COPY_AND_ASSIGN(EventPair);

 public:
  EventPair();

  ~EventPair() PADDLE_MAY_THROW;

  void Start(cudaStream_t stream);

  void Stop(cudaStream_t stream);

  void Reset();

  double Elapsed();

 private:
  cudaEvent_t start_;
  cudaEvent_t end_;
  bool started_;
};

class GPUEventTimer {
  DISABLE_COPY_AND_ASSIGN(GPUEventTimer);

 public:
  explicit GPUEventTimer(phi::GPUPlace place);

  void Start(cudaStream_t stream);

  void Stop(cudaStream_t stream);

  void Start();

  void Stop();

  void Reset();

  double Elapsed(bool reset);

  std::vector<double> ElapsedList(bool reset);

  void PreAlloc(size_t n);

  void ShrinkToFit();

  size_t Size() const;

  size_t Capacity() const;

 private:
  EventPair *GetLatest();

 private:
  std::vector<std::unique_ptr<EventPair>> events_;
  size_t length_;
  cudaStream_t default_stream_;
};

}  // namespace phi
