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

#include <brpc/channel.h>
#include <bthread/countdown_event.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace paddle {
namespace platform {

class RpcRequestStore {
 public:
  static RpcRequestStore& Instance() {
    static RpcRequestStore instance;
    return instance;
  }

  int GetRequestId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_id_ == INT32_MAX) {
      request_id_ = 0;
    } else {
      ++request_id_;
    }
    return request_id_;
  }

  std::shared_ptr<bthread::CountdownEvent> GetEvent(int request_id) {
    return id_to_event_map_[request_id];
  }

  std::string GetService(int request_id) {
    return id_to_service_map_[request_id];
  }

  std::string GetResponse(int request_id) {
    return id_to_resp_map_[request_id];
  }

  void InsertEvent(int request_id,
                   const std::shared_ptr<bthread::CountdownEvent>& event) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_event_map_.emplace(request_id, event);
  }

  void InsertService(int request_id, const std::string& service) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_service_map_.emplace(request_id, service);
  }

  void InsertResponse(int request_id, const std::string& resp) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_resp_map_.emplace(request_id, resp);
  }

 private:
  std::mutex mutex_;
  int request_id_;
  std::unordered_map<int, std::shared_ptr<bthread::CountdownEvent>>
      id_to_event_map_;
  std::unordered_map<int, std::string> id_to_resp_map_;
  std::unordered_map<int, std::string> id_to_service_map_;
};

int RpcSend(const std::string& service,
            const std::string& url,
            const std::string& query,
            void (*payload_builder)(brpc::Controller*, int, const std::string&),
            void (*response_handler)(brpc::Controller*,
                                     int,
                                     std::shared_ptr<bthread::CountdownEvent>),
            brpc::HttpMethod http_method = brpc::HttpMethod::HTTP_METHOD_POST,
            int timeout_ms = 10000,
            int max_retry = 3);

}  // namespace platform
}  // namespace paddle
