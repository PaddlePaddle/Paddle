//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <error.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

std::string getNcclVersion() {
  static std::once_flag ncclGetVersionFlag;
  static std::string versionString;

  std::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = platform::dynload::ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      auto ncclMajor = version / 1000;
      auto ncclMinor = (version % 1000) / 100;
      auto ncclPatch = version % (ncclMajor * 1000 + ncclMinor * 100);
      versionString = std::to_string(ncclMajor) + "." +
                      std::to_string(ncclMinor) + "." +
                      std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
         getNcclVersion();
}

const inline char* getNcclErrorDetailStr(
    ncclResult_t error, std::string processGroupFailureReason = "") {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != "") {
    return processGroupFailureReason.c_str();
  }
  switch (error) {
    case ncclUnhandledCudaError:
      return "ncclUnhandledCudaError: Call to CUDA function failed.";
    case ncclSystemError:
      return "ncclSystemError: System call (socket, malloc, munmap, etc) "
             "failed.";
    case ncclInternalError:
      return "ncclInternalError: Internal check failed. This is either a bug "
             "in NCCL or due to memory corruption";
    case ncclInvalidArgument:
      return "ncclInvalidArgument: Invalid value for an argument (such as "
             "invalid pointer, device count, ip:host pair, etc).";
    case ncclInvalidUsage:
      return "ncclInvalidUsage: This usually reflects invalid usage of NCCL "
             "library (such as too many async ops, too many collectives at "
             "once, mixing streams in a group, etc).";
    default:
      break;
  }
  return "Unknown NCCL error";
}

#define NCCL_CHECK(cmd, failureReason)                                    \
  do {                                                                    \
    ncclResult_t result = cmd;                                            \
    if (result != ncclSuccess) {                                          \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        ncclGetErrorWithVersion(result) + "\n" +          \
                        getNcclErrorDetailStr(result, failureReason);     \
    }                                                                     \
  } while (0)

#define NCCL_ASSERT(cmd)                                                \
  do {                                                                  \
    ncclResult_t result = cmd;                                          \
    if (result != ncclSuccess) {                                        \
      std::string err = ncclGetErrorWithVersion(result);                \
      fprintf(stderr, "NCCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      abort();                                                          \
    }                                                                   \
  } while (0)

class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm),
        aborted_(false),
        ncclAsyncErr_(ncclSuccess),
        commFailureReason_("") {}

  NCCLComm() : NCCLComm(nullptr) {}

  ~NCCLComm() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ncclComm_ && !aborted_) {
      NCCL_ASSERT(platform::dynload::ncclCommDestroy(ncclComm_));
    }
  }

  static std::shared_ptr<NCCLComm> create(int numRanks, int rank,
                                          ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();
    NCCL_CHECK(ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
               "");
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    return comm;
  }

  ncclUniqueId getNcclId() { return ncclId_; }

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  NCCLComm& operator=(NCCLComm&& other) = delete;

  // Move constructable
  NCCLComm(NCCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  }

  ncclComm_t getNcclComm();

  std::string getNcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void ncclCommAbort(std::string commFailureReason = "") {
    std::unique_lock<std::mutex> lock(mutex_);
    return;
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  ncclResult_t checkForNcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
    return ncclSuccess;
  }

 protected:
  ncclComm_t ncclComm_;
  ncclUniqueId ncclId_;
  bool aborted_;
  ncclResult_t ncclAsyncErr_;
  int rank_;
  mutable std::mutex mutex_;
  std::string commFailureReason_;
};

}  // namespace imperative
}  // namespace paddle
