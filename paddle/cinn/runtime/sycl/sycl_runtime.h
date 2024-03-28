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

#include <mutex>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "glog/logging.h"
#include "paddle/cinn/common/target.h"

using cinn::common::Target;

inline const char* SYCLGetErrorString(std::error_code error_code) {
  sycl::errc error_code_value = static_cast<sycl::errc>(error_code.value());
  switch (error_code_value) {
    case sycl::errc::success:
      return "SUCCESS";
    case sycl::errc::runtime:
      return "RUNTIME ERROR";
    case sycl::errc::kernel:
      return "KERNEL ERROR";
    case sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case sycl::errc::nd_range:
      return "NDRANGE ERROR";
    case sycl::errc::event:
      return "EVENT ERROR";
    case sycl::errc::kernel_argument:
      return "KERNEL ARGUMNET ERROR";
    case sycl::errc::build:
      return "BUILD ERROR";
    case sycl::errc::invalid:
      return "INVALID ERROR";
    case sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION";
    case sycl::errc::platform:
      return "PLATFORM ERROR";
    case sycl::errc::profiling:
      return "PROFILING ERROR";
    case sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case sycl::errc::kernel_not_supported:
      return "kERNEL NOT SUPPORTED";
    case sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";
    default:
      return "";
  }
}

/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CALL(func)                                                        \
  {                                                                            \
    try {                                                                      \
      func;                                                                    \
    } catch (const sycl::exception& e) {                                       \
      CHECK(e.code() == sycl::errc::success)                                   \
          << "SYCL Error, code="                                               \
          << ": " << SYCLGetErrorString(e.code()) << ", message:" << e.what(); \
      ;                                                                        \
    }                                                                          \
  }

/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace {
 public:
  // global platform
  std::vector<sycl::platform> platforms;
  // global platform name
  std::vector<std::string> platform_names;
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<sycl::device> devices;
  // the active devices id
  std::vector<int> active_device_ids;
  // the active contexts
  std::vector<sycl::context*> active_contexts;
  // the active queues
  std::vector<sycl::queue*> active_queues;
  // the events in active queues
  std::vector<std::vector<sycl::event>> active_events;
  // the mutex for initialization
  std::mutex mu;
  // destructor
  ~SYCLWorkspace() {
    for (auto queue : active_queues) {
      SYCL_CALL(queue->wait_and_throw());
      delete queue;
    }
    for (auto context : active_contexts) {
      delete context;
    }
  }
  // get the global workspace
  static SYCLWorkspace* Global();
  // Initialzie sycl devices.
  void Init(const Target::Arch arch, const std::string& platform_name = "");
  // set active devices
  void SetActiveDevices(std::vector<int> deviceIds);
  void* malloc(size_t nbytes, int device_id = 0);
  void free(void* data, int device_id = 0);
  void queueSync(int queue_id = 0);
  void memcpy(void* dest, const void* src, size_t nbytes, int queue_id = 0);
};
