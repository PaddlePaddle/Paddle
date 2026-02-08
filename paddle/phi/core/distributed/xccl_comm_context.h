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

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/comm_context.h"

#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/device_manager.h"

namespace phi {
class DenseTensor;
namespace distributed {

class XCCLCommContext final : public CommContext {
 public:
  XCCLCommContext(const phi::Place& place,
                  int rank,
                  int size,
                  const ccl::CCLRootId& xccl_id);
  ~XCCLCommContext();

  static void ReleaseAll();

  ccl::CCLComm GetXcclComm() const { return xccl_comm_; }

  std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }
  phi::stream::stream_t stream() const { return stream_->raw_stream(); }

  std::string GetDeviceType() const { return place_.GetDeviceType(); }

  phi::CustomContext* GetDevContext() { return dev_ctx_.get(); }

  void SetDevContext(std::unique_ptr<phi::CustomContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }

  void Broadcast(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 int root,
                 const phi::stream::stream_t& stream) const;

  void Send(const DenseTensor& in_tensor,
            const int64_t& count,
            const int& peer,
            const phi::stream::stream_t& stream) const;

  void Recv(DenseTensor* out_tensor,
            const int64_t& count,
            const int& peer,
            const phi::stream::stream_t& stream) const;

  void ReduceScatter(DenseTensor* out_tensor,
                     const DenseTensor& in_tensor,
                     phi::ccl::CCLReduceOp reduce_type,
                     const phi::stream::stream_t& stream) const;

  void AllGather(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 const phi::stream::stream_t& stream) const;

  void AllReduce(DenseTensor* out_tensor,
                 const DenseTensor& in_tensor,
                 phi::ccl::CCLReduceOp reduce_type,
                 const phi::stream::stream_t stream) const;

  void Reduce(DenseTensor* out_tensor,
              const DenseTensor& in_tensor,
              phi::ccl::CCLReduceOp reduce_type,
              int root,
              const phi::stream::stream_t& stream) const;

  void GroupStart() const;

  void GroupEnd() const;

 private:
  DISABLE_COPY_AND_ASSIGN(XCCLCommContext);

  phi::Place place_;
  ccl::CCLComm xccl_comm_;
  std::shared_ptr<phi::stream::Stream> stream_;
  std::unique_ptr<phi::CustomContext> dev_ctx_;
};

}  // namespace distributed
}  // namespace phi
#endif
