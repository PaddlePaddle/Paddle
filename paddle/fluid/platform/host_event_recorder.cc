/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/host_event_recorder.h"
#include "paddle/fluid/platform/os_info.h"

namespace paddle {
namespace platform {

ThreadEventRecorder::ThreadEventRecorder() {
  thread_id_ = ThreadIdRegistry::GetInstance().CurrentThreadId().MainTid();
  HostEventRecorder::GetInstance().RegisterThreadRecorder(thread_id_, this);
}

HostEventSection HostEventRecorder::GatherEvents() {
  HostEventSection host_sec;
  host_sec.thr_sections.reserve(thread_recorders_.size());
  for (auto &kv : thread_recorders_) {
    host_sec.thr_sections.emplace_back(std::move(kv.second->GatherEvents()));
  }
  return std::move(host_sec);
}

}  // namespace platform
}  // namespace paddle
