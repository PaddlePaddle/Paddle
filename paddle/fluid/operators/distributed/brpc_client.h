/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <time.h>

#include <chrono>  // NOLINT
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "brpc/channel.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace operators {
namespace distributed {

struct ChannelContext {
  brpc::Channel channel;
  std::shared_ptr<sendrecv::SendRecvService_Stub> stub;
};

typedef std::shared_ptr<ChannelContext> ChannelContextPtr;
typedef std::shared_ptr<framework::BlockingQueue<ChannelContextPtr>>
    ChannelQueuePtr;

class BRPCClient : public RPCClient {
 public:
  BRPCClient() {}
  virtual ~BRPCClient();

  bool AsyncSendVar(const std::string& ep, const platform::DeviceContext& ctx,
                    const framework::Scope& scope, const std::string& var_name,
                    int64_t time_out = RPCClient::rpc_time_out) override;

  bool AsyncGetVar(const std::string& ep, const platform::DeviceContext& ctx,
                   const framework::Scope& scope, const std::string& var_name,
                   int64_t time_out = RPCClient::rpc_time_out) override;

  bool AsyncPrefetchVar(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& in_var_name,
                        const std::string& out_var_name,
                        int64_t time_out = RPCClient::rpc_time_out) override;

  void AsyncSendBatchBarrier(
      const std::string& ep,
      int64_t time_out = RPCClient::rpc_time_out) override;

  void AsyncSendFetchBarrier(
      const std::string& ep,
      int64_t time_out = RPCClient::rpc_time_out) override;

  void Wait() override;

 private:
  void Proceed();
  ChannelQueuePtr GetChannel(const std::string& ep);

 private:
  std::unordered_map<std::string, ChannelQueuePtr> channels_;

  // mutex for Wait client sync
  std::mutex sync_mutex_;
  std::condition_variable sync_cond_;
  std::atomic<int64_t> req_count_{0};

  // mutex for GetChannel thread safety
  std::mutex chan_mutex_;
  DISABLE_COPY_AND_ASSIGN(BRPCClient);
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
