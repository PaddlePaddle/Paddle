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

/**
 * @brief MKLDNN CPU engine.
 *
 */
class CpuEngine {
public:
  static CpuEngine & Instance() {
    // Thread-safe in C++11.
    static CpuEngine myInstance;
    return myInstance;
  }

  // Disallow copy or move
  CpuEngine(const CpuEngine&) = delete;             // Copy constructor
  CpuEngine(CpuEngine&&) = delete;                  // Move constructor
  CpuEngine& operator=(const CpuEngine&) = delete;  // Copy assignment
  CpuEngine& operator=(CpuEngine&&) = delete;       // Move assignment

  mkldnn::engine & getEngine() { return cpuEngine_; }
protected:
  CpuEngine() : cpuEngine_(mkldnn::engine::cpu, 0) {}
//    CpuEngine() : cpuEngine_(mkldnn::engine::cpu_lazy, 0) {}
  ~CpuEngine() {}
private:
  mkldnn::engine cpuEngine_;
};

/**
 * @brief MKLDNN Stream.
 *
 */
class MkldnnStream {
public:
  MkldnnStream() : ready_(false) {
    resetState();
  }

  virtual ~MkldnnStream() {}

  /**
   * @brief Submit stream
   * @param prims The primitives vector
   *        block Waiting for the stream to complete
   */
  void submit(std::vector<mkldnn::primitive>& prims, bool block = true) {
    resetState();
    stream_->submit(prims).wait(block);
    ready_ = false;
  }

  /**
   * @brief Reset the mkldnn stream
   */
  void resetState() {
    if (ready_) {
      return;
    }
    // TODO(TJ): change me when mkldnn have method to reset this state
    stream_.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
//    stream_.reset(new stream(stream::kind::lazy));
    ready_ = true;
  }

private:
  bool ready_;
  std::shared_ptr<mkldnn::stream> stream_;
};

}  // namespace paddle
