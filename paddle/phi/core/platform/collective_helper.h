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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
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
  virtual gpuStream_t stream() const = 0;
  virtual gpuEvent_t compute_event() const = 0;
  virtual gpuEvent_t comm_event() const = 0;
  virtual phi::GPUContext* dev_context() const = 0;
  virtual ~NCCLComm() = default;
};

// A singleton NCCL communicator context reserves communication ring ids
class NCCLCommContext {
 public:
  static NCCLCommContext& Instance();

  NCCLComm* CreateComm(
      ncclUniqueId* nccl_id, int nranks, int rank, int dev_id, int ring_id = 0);

  void CreateAllNCCLComms(const std::vector<int>& dev_ids, int ring_id = 0);

  void CreateNCCLCommMultiTrainer(const std::vector<int>& dev_ids,
                                  ncclUniqueId* nccl_id,
                                  int nranks,
                                  int rank,
                                  int ring_id);

  // a latter comm with the same dev_id and the same ring_id
  // will override the former
  NCCLComm* AssignNCCLComm(
      ncclComm_t comm, int nranks, int rank, int dev_id, int ring_id = 0);

  // retrieve a communicator by the ring id in multiprocessing mode
  NCCLComm* Get(int ring_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator in ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_EQ(comm_map_.at(ring_id).size(),
                      1,
                      common::errors::InvalidArgument(
                          "One device id should be specified to retrieve from "
                          "multiple communicators."));
    return comm_map_.at(ring_id).begin()->second.get();
  }

  int GetRingId(ncclComm_t comm) const {
    for (const auto& pair : comm_map_) {
      for (const auto& p : pair.second) {
        if (p.second.get()->comm() == comm) {
          return pair.first;
        }
      }
    }
    return -1;
  }

  // retrieve a communicator by the ring id and the device id
  NCCLComm* Get(int ring_id, int dev_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator of ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_GT(
        comm_map_.at(ring_id).count(dev_id),
        0,
        common::errors::InvalidArgument(
            "Communicator at device id %d has not been initialized in ring %d.",
            dev_id,
            ring_id));
    return comm_map_.at(ring_id).at(dev_id).get();
  }

  // retrieve a communicator by the ring id and place
  NCCLComm* Get(int ring_id, Place place) const {
    return Get(ring_id, place.device);
  }

 private:
  std::once_flag once_flag_;
  std::mutex comm_map_mutex_;
  // ring id to dev-NCCLComm
  std::map<int, std::map<int, std::unique_ptr<NCCLComm>>> comm_map_;

  void ReleaseNCCLComms();

  NCCLCommContext() = default;
  DISABLE_COPY_AND_ASSIGN(NCCLCommContext);
};
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
// In order to apply hierarchical communication with BKCL, we need
// a communication ring contains BKCL communicators associated to a global
// BKCLUniqueId. E.g. for a hierarchical case,
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
// The BKCLComm instance is created and reversed in the BKCLCommContext
// singleton with a global user specified group id.
class BKCLComm {
 public:
  virtual int ring_id() const = 0;
  virtual int nranks() const = 0;
  virtual int rank() const = 0;
  virtual int device_id() const = 0;
  virtual BKCLContext_t comm() const = 0;
  virtual XPUStream stream() const = 0;
  virtual phi::XPUContext* dev_context() const = 0;
  virtual ~BKCLComm() = default;
};

// A singleton BKCL communicator context reserves communication ring ids
class BKCLCommContext {
 public:
  static BKCLCommContext& Instance() {
    static BKCLCommContext comm_ctx;
    return comm_ctx;
  }

  BKCLComm* CreateComm(
      BKCLUniqueId* bkcl_id, int nranks, int rank, int dev_id, int ring_id = 0);

  void CreateAllBKCLComms(const std::vector<int>& dev_ids, int ring_id = 0);

  // a latter comm with the same dev_id and the same ring_id
  // will override the former
  BKCLComm* AssignBKCLComm(
      BKCLContext_t comm, int nranks, int rank, int dev_id, int ring_id = 0);

  // retrieve a communicator by the ring id in multiprocessing mode
  BKCLComm* Get(int ring_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator in ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_EQ(comm_map_.at(ring_id).size(),
                      1,
                      common::errors::InvalidArgument(
                          "One device id should be specified to retrieve from "
                          "multiple communicators."));
    return comm_map_.at(ring_id).begin()->second.get();
  }

  // retrieve a communicator by the ring id and the device id
  BKCLComm* Get(int ring_id, int dev_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator of ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_GT(
        comm_map_.at(ring_id).count(dev_id),
        0,
        common::errors::InvalidArgument(
            "Communicator at device id %d has not been initialized in ring %d.",
            dev_id,
            ring_id));
    return comm_map_.at(ring_id).at(dev_id).get();
  }

  // retrieve a communicator by the ring id and place
  BKCLComm* Get(int ring_id, Place place) const {
    return Get(ring_id, place.device);
  }

 private:
  std::once_flag once_flag_;
  std::mutex comm_map_mutex_;
  // ring id to dev-BKCLComm
  std::map<int, std::map<int, std::unique_ptr<BKCLComm>>> comm_map_;

  void ReleaseBKCLComms();

  BKCLCommContext() = default;
  DISABLE_COPY_AND_ASSIGN(BKCLCommContext);
};
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
class XCCLComm {
 public:
  virtual int ring_id() const = 0;
  virtual int nranks() const = 0;
  virtual int rank() const = 0;
  virtual int device_id() const = 0;
  virtual phi::ccl::CCLComm comm() const = 0;
  virtual std::shared_ptr<phi::stream::Stream> stream() const = 0;
  virtual std::shared_ptr<phi::event::Event> compute_event() const = 0;
  virtual std::shared_ptr<phi::event::Event> comm_event() const = 0;
  virtual phi::CustomContext* dev_context() const = 0;
  virtual ~XCCLComm() = default;
};

// A singleton XCCL communicator context reserves communication ring ids
class XCCLCommContext {
 public:
  static XCCLCommContext& Instance(const std::string& device_type);
  static void Release();

  XCCLComm* CreateComm(phi::ccl::CCLRootId* nccl_id,
                       int nranks,
                       int rank,
                       int dev_id,
                       int ring_id = 0);

  void CreateAllXCCLComms(const std::vector<int>& dev_ids, int ring_id = 0);

  void CreateXCCLCommMultiTrainer(const std::vector<int>& dev_ids,
                                  phi::ccl::CCLRootId* xccl_id,
                                  int nranks,
                                  int rank,
                                  int ring_id);

  // a latter comm with the same dev_id and the same ring_id
  // will override the former
  XCCLComm* AssignXCCLComm(phi::ccl::CCLComm comm,
                           int nranks,
                           int rank,
                           int dev_id,
                           int ring_id = 0);

  // retrieve a communicator by the ring id in multiprocessing mode
  XCCLComm* Get(int ring_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator in ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_EQ(comm_map_.at(ring_id).size(),
                      1,
                      common::errors::InvalidArgument(
                          "One device id should be specified to retrieve from "
                          "multiple communicators."));
    return comm_map_.at(ring_id).begin()->second.get();
  }

  int GetRingId(phi::ccl::CCLComm comm) const {
    for (const auto& pair : comm_map_) {
      for (const auto& p : pair.second) {
        if (p.second.get()->comm() == comm) {
          return pair.first;
        }
      }
    }
    return -1;
  }

  // retrieve a communicator by the ring id and the device id
  XCCLComm* Get(int ring_id, int dev_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        common::errors::InvalidArgument(
            "Communicator of ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_GT(
        comm_map_.at(ring_id).count(dev_id),
        0,
        common::errors::InvalidArgument(
            "Communicator at device id %d has not been initialized in ring %d.",
            dev_id,
            ring_id));
    return comm_map_.at(ring_id).at(dev_id).get();
  }

  // retrieve a communicator by the ring id and place
  XCCLComm* Get(int ring_id, Place place) const {
    return Get(ring_id, place.device);
  }

 private:
  std::string device_type_;
  std::once_flag once_flag_;
  std::mutex comm_map_mutex_;
  // ring id to dev-XCCLComm
  std::map<int, std::map<int, std::unique_ptr<XCCLComm>>> comm_map_;

  void ReleaseXCCLComms();

  XCCLCommContext() = default;
  explicit XCCLCommContext(const std::string& device_type)
      : device_type_(device_type) {}
  DISABLE_COPY_AND_ASSIGN(XCCLCommContext);
};
#endif
}  // namespace platform
}  // namespace paddle
