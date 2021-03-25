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

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {

int CreateListenSocket(const std::string& ep);

void CloseSocket(int fd);

void SendBroadCastNCCLID(std::vector<std::string> servers, int nccl_comm_num,
                         std::function<std::string(size_t)> func,
                         const framework::Scope& scope);

// server listen on endpoint, then recv nccl id
void RecvBroadCastNCCLID(std::string endpoint, int nccl_comm_num,
                         std::function<std::string(size_t)> func,
                         const framework::Scope& scope);

// recv nccl id from socket
template <typename CommUniqueId>
void RecvBroadCastCommID(int server_fd, std::string endpoint,
                         std::vector<CommUniqueId>* nccl_ids);

class SocketServer {
 public:
  SocketServer() = default;

  ~SocketServer() { CloseSocket(server_fd_); }

  int socket() const { return server_fd_; }

  static SocketServer& GetInstance(const std::string& end_point);

 private:
  int server_fd_{-1};
  std::string end_point_;

  static std::once_flag init_flag_;
};

}  // namespace platform
}  // namespace paddle
