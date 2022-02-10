/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL)
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "glog/logging.h"

namespace paddle {
namespace platform {

int CreateListenSocket(const std::string& ep);

void CloseSocket(int fd);

template <typename CommUniqueId>
void SendBroadCastCommID(std::vector<std::string> servers,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id = 0);

template <typename CommUniqueId>
void RecvBroadCastCommID(std::string endpoint,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id = 0);

// recv nccl id from socket
template <typename CommUniqueId>
void RecvBroadCastCommID(int server_fd, std::string endpoint,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id = 0);

class SocketServer {
 public:
  SocketServer() = default;

  ~SocketServer() {
    if (server_fd_ != -1) {
      CloseSocket(server_fd_);
    }
  }

  int socket() const { return server_fd_; }

  void Release() {
    VLOG(3) << "Server will be closed by external call.";
    CloseSocket(server_fd_);
    server_fd_ = -1;
  }

  static SocketServer& GetInstance(const std::string& end_point);

 private:
  int server_fd_{-1};
  std::string end_point_;

  static std::once_flag init_flag_;
};

}  // namespace platform
}  // namespace paddle

#endif
