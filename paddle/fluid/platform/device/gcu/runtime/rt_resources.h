/* Copyright (c) 2023 Enflame. All Rights Reserved.

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

#include <map>
#include <memory>
#include <string>

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"
#include "paddle/fluid/platform/device/gcu/utils/cache.h"

namespace paddle {
namespace distributed {
class ProcessGroupCustom;
}
}  // namespace paddle

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

using ProcessGroupCustom = paddle::distributed::ProcessGroupCustom;
using ContextCaches = paddle::platform::gcu::Cache<uint32_t, Context>;
using ProcessGroupCaches =
    paddle::platform::gcu::Cache<uint32_t, ProcessGroupCustom>;
using RuntimeInfoCaches =
    paddle::platform::gcu::Cache<uint32_t, GcuRunTimeInfo>;

class ResourceMgr {
 public:
  // static ResourceMgr& GetInstance() {
  //   static ResourceMgr rm;
  //   return rm;
  // }
  static ResourceMgr* GetInstance();

  ContextCaches* GetContextCaches() { return context_caches_.get(); }

  ProcessGroupCaches* GetProcessGroupCaches() { return pg_caches_.get(); }

  RuntimeInfoCaches* GetRuntimeInfoCaches() { return rt_info_caches_.get(); }

  void SetCurrentDevice(int device_id) { current_device_id_ = device_id; }

  int GetCurrentDevice() { return current_device_id_; }

  std::string GetRTMetricsReport();

  void RTCounter(const std::string& key, int64_t value);

  std::string RTMemoryUseInfo(const Context* ctx);

  ~ResourceMgr();

 private:
  ResourceMgr();

  RT_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);

 private:
  std::unique_ptr<ContextCaches> context_caches_;
  std::unique_ptr<ProcessGroupCaches> pg_caches_;
  std::unique_ptr<RuntimeInfoCaches> rt_info_caches_;
  std::map<std::string, int64_t> rt_metrics_;
  int current_device_id_ = 0;
  std::mutex lock_;  // Multi threading support in the future
};

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
