/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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
  MKLDNN_BASE = 1,   // basical info of MKLDNN
  MKLDNN_TESTS = 1,  // gtest info of MKLDNN
  MKLDNN_FMTS = 2,   // format info of MKLDNN
  MKLDNN_SIZES = 3,  // size info of MKLDNN
  MKLDNN_ALL = 4,    // show all info of MKLDNN
} MKLDNN_LOG_LEVEL;

/**
 * @brief MKLDNN CPU engine.
 *
 */
class CPUEngine {
 public:
  static CPUEngine& Instance() {
    // Thread-safe in C++11.
    static CPUEngine myInstance;
    return myInstance;
  }

  // Disallow copy or move
  CPUEngine(const CPUEngine&) = delete;             // Copy constructor
  CPUEngine(CPUEngine&&) = delete;                  // Move constructor
  CPUEngine& operator=(const CPUEngine&) = delete;  // Copy assignment
  CPUEngine& operator=(CPUEngine&&) = delete;       // Move assignment

  mkldnn::engine& getEngine() { return cpuEngine_; }

 protected:
  CPUEngine() : cpuEngine_(mkldnn::engine::cpu, 0) {}
  //    CPUEngine() : cpuEngine_(mkldnn::engine::cpu_lazy, 0) {}
  ~CPUEngine() {}

 private:
  mkldnn::engine cpuEngine_;
};

/**
 * @brief MKLDNN Stream.
 *
 */
class MKLDNNStream {
 public:
  MKLDNNStream() : ready_(false) { resetState(); }

  virtual ~MKLDNNStream() {}

  /**
   * @brief Submit stream
   * @param prims The primitives vector
   * @param block Waiting for the stream to complete
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
    // stream_.reset(new mkldnn::stream(mkldnn::stream::kind::lazy));
    stream_.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    ready_ = true;
  }

 private:
  bool ready_;
  std::shared_ptr<mkldnn::stream> stream_;
};

}  // namespace paddle
