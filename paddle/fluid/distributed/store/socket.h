// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PADDLE_FLUID_DISTRIBUTED_SOCKET_SOCKET_H_
#define PADDLE_FLUID_DISTRIBUTED_SOCKET_SOCKET_H_

#include <netdb.h>
#include <chrono>
#include <iostream>
#include <string>

namespace paddle {
namespace distributed {

class SocketOptions {
 public:
  void set_connect_timeout(std::chrono::milliseconds timeout) {
    _connect_timeout = timeout;
  }
  std::chrono::milliseconds get_connect_timeout() const {
    return _connect_timeout;
  }

 private:
  std::chrono::milliseconds _connect_timeout{30};
};

class Socket {
 public:
  Socket() = default;
  ~Socket() = default;

  static Socket listen(std::uint16_t port, const SocketOptions& opts = {});
  static Socket connect(const std::string& host, std::uint16_t port,
                        const SocketOptions& opts = {});
  Socket accept() const;
  explicit Socket(int sock) : _sock(sock) {}

  // template <typename T>
  // void sendValue(const T& value);
  // template <typename T>
  // T recvValue();

  std::uint16_t getPort() const;
  int get_socket_fd() const { return _sock; }

 private:
  int _sock;

  // template <typename T>
  // void sendBytes(const T* buffer, size_t len);
  // template <typename T>
  // void recvBytes(T* buffer, size_t len);
};

}  // namespace distributed
}  // namespace paddle

#endif  // PADDLE_FLUID_DISTRIBUTED_SOCKET_SOCKET_H_
