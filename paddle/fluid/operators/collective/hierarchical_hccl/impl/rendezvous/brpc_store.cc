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

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/brpc_store.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {
namespace rendezvous {

DEFINE_int32(hierarchical_hccl_rpc_retry_times, 0,
             "Specify rpc max try times!");
DECLARE_int32(hierarchical_hccl_rpc_retry_times);

void BRPCStore::set(const std::string &key, const std::string &data) {
  paddle::operators::rendezvous::proto::PutKVRequest request;
  request.set_key(key);
  request.set_value(std::string(data.begin(), data.end()));
  paddle::operators::rendezvous::proto::PutKVResponse response;
  std::function<void(brpc::Controller *,
                     const paddle::operators::rendezvous::proto::PutKVRequest *,
                     paddle::operators::rendezvous::proto::PutKVResponse *,
                     google::protobuf::Closure *)>
      method =
          [=](brpc::Controller *cntl,
              const paddle::operators::rendezvous::proto::PutKVRequest *req,
              paddle::operators::rendezvous::proto::PutKVResponse *res,
              google::protobuf::Closure *done) {
            _stub->PutKV(cntl, req, res, done);
          };
  std::function<bool(paddle::operators::rendezvous::proto::PutKVResponse *)>
      checker = [=](paddle::operators::rendezvous::proto::PutKVResponse *res) {
        return res->status() ==
               paddle::operators::rendezvous::proto::Status::OK;
      };

  VLOG(1) << "try to set key[" << key << "] to rendezvous server...";
  perform_rpc<paddle::operators::rendezvous::proto::PutKVRequest,
              paddle::operators::rendezvous::proto::PutKVResponse>(
      &request, &response, method, checker);
  VLOG(1) << "successfully set [" << key << "] to rendezvous server.";
}

std::string BRPCStore::get(const std::string &key) {
  paddle::operators::rendezvous::proto::GetValueRequest request;
  request.set_key(key);
  paddle::operators::rendezvous::proto::GetValueResponse response;

  std::function<void(
      brpc::Controller *,
      const paddle::operators::rendezvous::proto::GetValueRequest *,
      paddle::operators::rendezvous::proto::GetValueResponse *,
      google::protobuf::Closure *)>
      method =
          [=](brpc::Controller *cntl,
              const paddle::operators::rendezvous::proto::GetValueRequest *req,
              paddle::operators::rendezvous::proto::GetValueResponse *res,
              google::protobuf::Closure *done) {
            _stub->GetValue(cntl, req, res, done);
          };
  std::function<bool(paddle::operators::rendezvous::proto::GetValueResponse *)>
      checker =
          [=](paddle::operators::rendezvous::proto::GetValueResponse *res) {
            return res->status() ==
                   paddle::operators::rendezvous::proto::Status::OK;
          };

  VLOG(1) << "requesting key [" << key << "] from rendezvous server...";
  perform_rpc(&request, &response, method, checker);
  VLOG(1) << "got key[" << key << "]";

  return response.value();
}

template <typename R, typename S>
void BRPCStore::perform_rpc(
    const R *request, S *response,
    std::function<void(brpc::Controller *, const R *, S *,
                       google::protobuf::Closure *)>
        method,
    std::function<bool(S *)> checker) {
  brpc::Controller cntl;
  int retry_times = FLAGS_hierarchical_hccl_rpc_retry_times <= 0
                        ? MAX_RETRY_TIMES
                        : FLAGS_hierarchical_hccl_rpc_retry_times;

  for (auto retry_cnt = 0; retry_cnt < retry_times; ++retry_cnt) {
    try {
      VLOG(1) << "before making brpc request...";
      cntl.Reset();
      method(&cntl, request, response, nullptr);
      if (!cntl.Failed() && checker(response)) {
        VLOG(1) << "rpc call succeeded...";
        return;
      } else {
        if (cntl.Failed()) {
          VLOG(1) << "rpc server not ready or died, can retry, " << retry_cnt
                  << "/" << retry_times;
        } else {
          VLOG(1) << "rpc call failed, can retry, " << retry_cnt << "/"
                  << retry_times;
        }
      }
    } catch (std::exception &e) {
      VLOG(1) << "Exception: " << e.what();
    }
    // sleep for 500ms before another try.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(RETRY_WAITING_TIME_MILLSEC));
  }

  PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
      "BRPC call failed too many times, aborting. "
      "Please set env FLAGS_hierarchical_hccl_rpc_retry_times"));
}

}  // namespace rendezvous
}  // namespace operators
}  // namespace paddle
