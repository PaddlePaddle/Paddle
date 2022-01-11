/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/profiler.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/cuda_tracer.h"
#include "paddle/fluid/platform/profiler/host_tracer.h"

namespace paddle {
namespace platform {

std::atomic<bool> Profiler::alive_{false};

std::unique_ptr<Profiler> Profiler::Create(const ProfilerOptions& options) {
  if (alive_.exchange(true)) {
    return nullptr;
  }
  return std::unique_ptr<Profiler>(new Profiler(options));
}

Profiler::~Profiler() { alive_.store(false); }

Profiler::Profiler(const ProfilerOptions& options) {
  tracers_.emplace_back(new HostTracer(), true);
  tracers_.emplace_back(&CudaTracer::GetInstance(), false);
}

}  // namespace platform
}  // namespace paddle
