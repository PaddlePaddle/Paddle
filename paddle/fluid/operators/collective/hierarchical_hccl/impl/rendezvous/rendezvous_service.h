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

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/rendezvous.pb.h"

#include <brpc/server.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {
namespace rendezvous {

class RendezvousServiceImpl
    : public paddle::operators::rendezvous::proto::RendezvousService {
 public:
  RendezvousServiceImpl() {}
  virtual ~RendezvousServiceImpl() {}
  virtual void PutKV(
      google::protobuf::RpcController *cntl_base,
      const paddle::operators::rendezvous::proto::PutKVRequest *request,
      paddle::operators::rendezvous::proto::PutKVResponse *response,
      google::protobuf::Closure *done) {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> g(_record_lock);

    auto got = _records.find(request->key());

    if (got != _records.end()) {
      VLOG(1) << "A repeated key [" << request->key()
              << "], try to replace it!";
      _records.erase(request->key());
    }

    _records.emplace(request->key(), request->value());
    response->set_status(paddle::operators::rendezvous::proto::Status::OK);
  }

  virtual void GetValue(
      google::protobuf::RpcController *cntl_base,
      const paddle::operators::rendezvous::proto::GetValueRequest *request,
      paddle::operators::rendezvous::proto::GetValueResponse *response,
      google::protobuf::Closure *done) {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> g(_record_lock);
    if (_records.find(request->key()) == _records.end()) {
      response->set_status(
          paddle::operators::rendezvous::proto::Status::NOT_FOUND);
    } else {
      response->set_status(paddle::operators::rendezvous::proto::Status::OK);
      response->set_value(_records[request->key()]);
    }
  }

  void clear_group(std::string group_name) { _records.clear(); }

 private:
  std::unordered_map<std::string, std::string> _records;
  std::mutex _record_lock;
};

}  // namespace rendezvous
}  // namespace operators
}  // namespace paddle
