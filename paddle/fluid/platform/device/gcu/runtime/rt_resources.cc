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

#include "paddle/fluid/platform/device/gcu/runtime/rt_resources.h"

#include <dtu/dtu_sdk/dtu_sdk.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <string>

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

std::atomic<ResourceMgr*> g_resource_mgr(nullptr);
std::once_flag g_resource_mgr_once;

ResourceMgr::ResourceMgr()
    : context_caches_(new ContextCaches()),
      pg_caches_(new ProcessGroupCaches()),
      rt_info_caches_(new RuntimeInfoCaches()) {
  ::dtu::DTUSDKInit();
}

ResourceMgr::~ResourceMgr() {
  context_caches_.reset();
  pg_caches_.reset();
  rt_info_caches_.reset();
  ::dtu::DTUSDKFini();
}

ResourceMgr* ResourceMgr::GetInstance() {
  std::call_once(g_resource_mgr_once, [&]() {
    auto rm_ptr = std::unique_ptr<ResourceMgr>(new ResourceMgr());
    g_resource_mgr = rm_ptr.release();
  });
  return g_resource_mgr.load();
}

std::string ResourceMgr::GetRTMetricsReport() {
  std::lock_guard<std::mutex> slock(lock_);
  std::stringstream ss;
  ss << "Gcu Runtime Use Metrics Report\n";
  for (auto it = rt_metrics_.begin(); it != rt_metrics_.end(); ++it) {
    ss << "Counter: " << it->first << "\n";
    ss << "  Value: " << it->second << "\n";
  }
  return ss.str();
}

void ResourceMgr::RTCounter(const std::string& key, int64_t value) {
  std::lock_guard<std::mutex> slock(lock_);
  if (rt_metrics_.find(key) != rt_metrics_.end()) {
    rt_metrics_[key] += value;
  } else {
    rt_metrics_[key] = value;
  }
}

std::string ResourceMgr::RTMemoryUseInfo(const Context* ctx) {
  std::lock_guard<std::mutex> slock(lock_);
  std::stringstream mem_info;
  auto ctx_name = ctx->GetName();
  mem_info << ctx_name << ": mem alive = ";
  std::string key = ctx_name + "_Memory";
  if (rt_metrics_.find(key) != rt_metrics_.end()) {
    mem_info << rt_metrics_[key];
  } else {
    mem_info << 0;
  }
  mem_info << ", mem use = ";
  key = ctx_name + "_MemoryUse";
  if (rt_metrics_.find(key) != rt_metrics_.end()) {
    mem_info << rt_metrics_[key] / 1024.0 / 1024.0;
  } else {
    mem_info << 0;
  }
  mem_info << "Mbs.";
  return mem_info.str();
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
