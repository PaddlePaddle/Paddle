/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "CustomStackTrace.h"
#include <gflags/gflags.h>
#include <iostream>

DEFINE_bool(
    layer_stack_error_only_current_thread,
    true,
    "Dump current thread or whole process layer stack when signal error "
    "occurred. true means only dump current thread layer stack");

namespace paddle {

CustomStackTrace<std::string> gLayerStackTrace;

static std::mutex gLayerStackTraceMtx;
void installLayerStackTracer() {
  logging::installFailureWriter([](const char* data, int sz) {
    std::lock_guard<std::mutex> guard(gLayerStackTraceMtx);
    if (!gLayerStackTrace.empty()) {
      size_t curTid = -1UL;
      std::hash<std::thread::id> hasher;
      gLayerStackTrace.dump(
          [&curTid, &hasher](std::thread::id tid,
                             bool* isForwarding,
                             const std::string& layerName) {
            if (curTid != hasher(tid)) {
              if (curTid != -1UL) {
                std::cerr << std::endl;
              }
              curTid = hasher(tid);
              std::cerr << "Thread [" << tid << "] ";
              if (isForwarding) {
                std::cerr << (*isForwarding ? "Forwarding " : "Backwarding ");
              }
            }
            std::cerr << layerName << ", ";
          },
          FLAGS_layer_stack_error_only_current_thread);
      std::cerr << std::endl;
    }
    std::cerr.write(data, sz);
  });
}

}  // namespace paddle
