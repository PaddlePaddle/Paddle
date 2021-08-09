/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <brpc/channel.h>

#include <thread>

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/brpc_store.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/rendezvous.pb.h"

namespace paddle {
namespace operators {
namespace rendezvous {

#define MAX_RETRY_TIMES 30
#define RETRY_WAITING_TIME_MILLSEC 500

class BRPCStore {
 public:
  explicit BRPCStore(std::string service_endpoint) {
    CHECK_EQ(_channel.Init(service_endpoint.c_str(), "", &_options), 0);
    _stub.reset(
        new paddle::operators::rendezvous::proto::RendezvousService_Stub(
            &_channel));
  }

  ~BRPCStore() { _stub.release(); }

  void set(const std::string &key, const std::string &data);
  std::string get(const std::string &key);

 private:
  template <typename R, typename S>
  void perform_rpc(const R *request, S *response,
                   std::function<void(brpc::Controller *, const R *, S *,
                                      google::protobuf::Closure *)>
                       method,
                   std::function<bool(S *)> checker);

 private:
  brpc::Channel _channel;
  brpc::ChannelOptions _options;
  std::unique_ptr<paddle::operators::rendezvous::proto::RendezvousService_Stub>
      _stub;

  DISALLOW_COPY_AND_ASSIGN(BRPCStore);
};

}  // namespace rendezvous
}  // namespace operators
}  // namespace paddle
