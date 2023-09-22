// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/macros.h"

#include "paddle/phi/backends/device_manager.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "xpu/bkcl.h"
#endif
namespace phi {
class DenseTensor;
namespace distributed {

class XCCLCommContext final : public CommContext {
 public:
  XCCLCommContext(const std::string& device_type,
                  int rank,
                  int size,
                  const ccl::CCLRootId& xccl_id);

  ccl::CCLComm GetXcclComm() const { return xccl_comm_; }

  const std::string& GetDeviceType() const { return device_type_; }

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root,
                 const phi::stream::Stream& stream) const;

  void Send(const phi::DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            const phi::stream::Stream& stream) const;

  void Recv(phi::DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            const phi::stream::Stream& stream) const;

  void ReduceScatter(phi::DenseTensor* out_tensor,
                     const phi::DenseTensor& in_tensor,
                     phi::ccl::CCLReduceOp reduce_type,
                     const phi::stream::Stream& stream) const;

  void AllGather(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 const phi::stream::Stream& stream) const;

  void AllReduce(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 phi::ccl::CCLReduceOp reduce_type,
                 const phi::stream::Stream& stream) const;

  void Reduce(phi::DenseTensor* out_tensor,
              const phi::DenseTensor& in_tensor,
              phi::ccl::CCLReduceOp reduce_type,
              int root,
              const phi::stream::Stream& stream) const;

  void GroupStart() const;

  void GroupEnd() const;

 private:
  DISABLE_COPY_AND_ASSIGN(XCCLCommContext);

  std::string device_type_;
  ccl::CCLComm xccl_comm_;
};

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
        phi::errors::InvalidArgument(
            "Communicator in ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_EQ(comm_map_.at(ring_id).size(),
                      1,
                      phi::errors::InvalidArgument(
                          "One device id should be specified to retrieve from "
                          "multiple communicators."));
    return comm_map_.at(ring_id).begin()->second.get();
  }

  // retrieve a communicator by the ring id and the device id
  BKCLComm* Get(int ring_id, int dev_id) const {
    PADDLE_ENFORCE_GT(
        comm_map_.count(ring_id),
        0,
        phi::errors::InvalidArgument(
            "Communicator of ring id %d has not been initialized.", ring_id));
    PADDLE_ENFORCE_GT(
        comm_map_.at(ring_id).count(dev_id),
        0,
        phi::errors::InvalidArgument(
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

}  // namespace distributed
}  // namespace phi
