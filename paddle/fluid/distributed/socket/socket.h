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

#pragma once
#include <netdb.h>
#include <iostream>
#include <string>

#include "paddle/fluid/distributed/socket/tcp_utils.h"

namespace paddle {
namespace distributed {

class SocketOptions {
 public:
  void set_connect_timeout(std::chrono::seconds timeout) {
    _connect_timeout = timeout;
  }
  std::chrono::seconds get_connect_timeout const { return _connect_timeout; }

 private:
  std::chrono::seconds _connect_timeout{30};
};

class Socket {
 public:
  Socket() = default;
  Socket(const Socket& o) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(const Socket&& o);
  Socket& operator=(const Socket&&);
  ~Socket() {}

  static Socket& listen(std::uint16_t port, const SocketOptions& opts = {});
  static Socket& connect(const std::string& host, std::uint16_t port,
                         const SocketOptions& opts = {});
  Socket accept() const;

  template <typename T>
  void sendValue(const T value);
  template <typename T>
  T recvValue();

  std::uint16_t getPort() const;

 private:
  explicit Socket(int sock) : _sock(sock) {}
  int _sock;

  template <typename T>
  void sendBytes(const T* buffer, size_t len);
  template <typename T>
  void recvBytes(T* buffer, size_t len);
}

}  // namespace distributed
}  // namespace paddle
