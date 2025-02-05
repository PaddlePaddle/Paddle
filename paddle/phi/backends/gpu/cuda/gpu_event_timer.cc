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

#include "paddle/phi/backends/gpu/cuda/gpu_event_timer.h"
#include <numeric>
#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

EventPair::EventPair() {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&start_));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&end_));
  started_ = false;
}

EventPair::~EventPair() PADDLE_MAY_THROW {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(start_));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(end_));
}

void EventPair::Start(cudaStream_t stream) {
  PADDLE_ENFORCE_EQ(
      started_,
      false,
      phi::errors::InvalidArgument("GPUEventTimer has been started."));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(start_, stream));
  started_ = true;
}

void EventPair::Stop(cudaStream_t stream) {
  PADDLE_ENFORCE_EQ(
      started_,
      true,
      phi::errors::InvalidArgument("GPUEventTimer has not been started."));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(end_, stream));
  started_ = false;
}

void EventPair::Reset() { started_ = false; }

double EventPair::Elapsed() {
  PADDLE_ENFORCE_EQ(
      started_,
      false,
      phi::errors::InvalidArgument("GPUEventTimer has not been stopped."));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(start_));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(end_));
  float ms;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(&ms, start_, end_));
  return ms / 1000.0;
}

GPUEventTimer::GPUEventTimer(phi::GPUPlace place) : length_(0) {
  auto *dev_ctx = phi::DeviceContextPool::Instance().GetByPlace(place);
  default_stream_ = dev_ctx->stream();
}

EventPair *GPUEventTimer::GetLatest() {
  PADDLE_ENFORCE_GT(
      length_,
      0,
      phi::errors::InvalidArgument("GPUEventTimer has not been started."));
  auto &back = events_[length_ - 1];
  if (back == nullptr) {
    back.reset(new EventPair());
  }
  return back.get();
}

void GPUEventTimer::Start(cudaStream_t stream) {
  if (length_ == events_.size()) {
    VLOG(10) << "Expand when length = " << length_;
    events_.emplace_back();
  }
  ++length_;
  GetLatest()->Start(stream);
}

void GPUEventTimer::Stop(cudaStream_t stream) { GetLatest()->Stop(stream); }

void GPUEventTimer::Start() { Start(default_stream_); }

void GPUEventTimer::Stop() { Stop(default_stream_); }

void GPUEventTimer::Reset() {
  for (size_t i = 0; i < length_; ++i) {
    events_[i]->Reset();
  }
  length_ = 0;
}

double GPUEventTimer::Elapsed(bool reset) {
  double ret = 0;
  for (size_t i = 0; i < length_; ++i) {
    ret += events_[i]->Elapsed();
  }
  if (reset) {
    Reset();
  }
  return ret;
}

std::vector<double> GPUEventTimer::ElapsedList(bool reset) {
  std::vector<double> values(length_);
  for (size_t i = 0; i < length_; ++i) {
    values[i] = events_[i]->Elapsed();
  }
  if (reset) {
    Reset();
  }
  return values;
}

void GPUEventTimer::PreAlloc(size_t n) {
  if (events_.size() >= n) return;
  events_.resize(n);
  for (auto &pair : events_) {
    if (pair == nullptr) {
      pair.reset(new EventPair());
    }
  }
}

void GPUEventTimer::ShrinkToFit() { events_.resize(length_); }

size_t GPUEventTimer::Size() const { return length_; }

size_t GPUEventTimer::Capacity() const { return events_.size(); }

}  // namespace phi
