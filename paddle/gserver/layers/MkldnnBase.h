/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

#include "mkldnn.hpp"

namespace paddle {

typedef enum {
  DNN_BASE = 1,
  DNN_TESTS = 1,
  DNN_SIZES,
  DNN_FMTS,
  DNN_TESTS_DETAILS,
  DNN_TESTS_MORE,
  DNN_ALL,
} DNN_LOG_LEVEL;

/// For mkldnn cpu engine
class CpuEngine {
public:
  static CpuEngine & Instance() {
    // I's thread-safe in C++11.
    static CpuEngine myInstance;
    return myInstance;
  }
  CpuEngine(CpuEngine const&) = delete;             // Copy construct
  CpuEngine(CpuEngine&&) = delete;                  // Move construct
  CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
  CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

  mkldnn::engine & getEngine() { return cpuEngine_; }
protected:
  CpuEngine() : cpuEngine_(mkldnn::engine::cpu, 0) {}
//    CpuEngine() : cpuEngine_(mkldnn::engine::cpu_lazy, 0) {}
  ~CpuEngine() {}
private:
  mkldnn::engine cpuEngine_;
};

}  // namespace paddle
