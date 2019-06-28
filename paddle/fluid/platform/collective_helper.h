//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

// In order to apply hierarchical communication with NCCL, we need
// a communication ring contains NCCL communicators associated to a global
// ncclUniqueId. E.g. for a hierarchical case,
//
//    11 - 12   21 - 22
//     |    |    |    |
//    13 - 14 - 23 - 24
//          |    |
//    31 - 32 - 41 - 42
//     |    |    |    |
//    33 - 34   43 - 44
//
// we group (14,23,32,41) as the top, and (11,12,13,14), (21,22,23,24),
// (31,32,33,34), (41,42,43,44) as bottoms respectively.
//
// We could also use a single communication ring for the flatten case
//
// The NCCLComm instance is created and reversed in the NCCLCommContext
// singleton with a global user specified group id.
class NCCLComm {
 public:
  virtual int ring_id() const = 0;
  virtual int nranks() const = 0;
  virtual int rank() const = 0;
  virtual int device_id() const = 0;
  virtual ncclComm_t comm() const = 0;
  virtual cudaStream_t stream() const = 0;
  virtual ~NCCLComm() = default;
};

// a singleton NCCL communicator context reserves communication ring ids
// Assume multiprocessing mode
class NCCLCommContext {
 public:
  static NCCLCommContext& Instance() {
    static NCCLCommContext comm_ctx;
    return comm_ctx;
  }
  ~NCCLCommContext();

  NCCLComm* CreateNCCLComm(ncclUniqueId* nccl_id, int nranks, int rank,
                           int dev_id, int ring_id = 0);

  // retrieve a communicator by the ring id
  NCCLComm* Get(int ring_id) const {
    PADDLE_ENFORCE(comm_map_.count(ring_id),
                   "comunicator in ring id %d has not been initialized",
                   ring_id);
    return comm_map_.at(ring_id).get();
  }

 private:
  // ring id to NCCLComm
  std::unordered_map<int, std::unique_ptr<NCCLComm>> comm_map_;

  // device id to CUDADeviceContext
  std::unordered_map<int, std::unique_ptr<CUDADeviceContext>> dev_ctx_map_;

  NCCLCommContext() = default;
  NCCLCommContext(const NCCLCommContext& other) = delete;
  NCCLCommContext& operator=(const NCCLCommContext& other) = delete;
};

}  // namespace platform
}  // namespace paddle

#endif
