// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <chrono>
#include <iostream>
#include <vector>

#include "paddle/phi/core/enforce.h"

// Utility functions for TCP socket.
namespace phi {
namespace distributed {

#ifdef _WIN32
using SocketType = SOCKET;
#else
using SocketType = int;
#endif

namespace tcputils {

constexpr int LISTENQ = 2048;
constexpr std::chrono::seconds kDelay = std::chrono::seconds(3);
constexpr std::chrono::seconds kNoTimeout = std::chrono::seconds::zero();
constexpr std::chrono::seconds kDefaultTimeout = std::chrono::seconds(360);

std::error_code socket_error();
void close_socket(SocketType socket);
::addrinfo* get_addr_info(const std::string host,
                          const std::string port,
                          int ai_flags,
                          int family);
void free_addr_info(::addrinfo*);
SocketType tcp_connect(const std::string host,
                       const std::string port,
                       int family,
                       std::chrono::seconds timeout = kNoTimeout);
SocketType tcp_listen(const std::string host,
                      const std::string port,
                      int family);
SocketType tcp_accept(SocketType socket);

void send_string(SocketType socket, const std::string& s);
std::string receive_string(SocketType socket);

template <typename T>
void send_bytes(SocketType socket, const T* buffer, size_t len) {
  size_t to_send = len * sizeof(T);
  if (to_send == 0) {
    return;
  }

  auto ptr = reinterpret_cast<const char*>(buffer);

  while (to_send > 0) {
    auto byte_sent = ::send(socket, ptr, to_send, 0);
    PADDLE_ENFORCE_GT(
        byte_sent,
        0,
        phi::errors::InvalidArgument("TCP send error. Details: %s.",
                                     socket_error().message()));
    to_send -= byte_sent;
    ptr += byte_sent;
  }
}

template <typename T>
void receive_bytes(SocketType socket, T* buffer, size_t len) {
  size_t to_recv = len * sizeof(T);
  if (to_recv == 0) {
    return;
  }
  auto ptr = reinterpret_cast<char*>(buffer);

  while (to_recv > 0) {
    auto byte_received = ::recv(socket, ptr, to_recv, 0);
    PADDLE_ENFORCE_GT(
        byte_received,
        0,
        phi::errors::InvalidArgument("TCP receive error. Details: %s.",
                                     socket_error().message()));

    to_recv -= byte_received;
    ptr += byte_received;
  }
}

template <typename T>
void send_vector(SocketType socket, const std::vector<T>& v) {
  size_t size = v.size();
  send_bytes<size_t>(socket, &size, 1);
  send_bytes<T>(socket, v.data(), size);
}

template <typename T>
std::vector<T> receive_vector(SocketType socket) {
  size_t size;
  receive_bytes<size_t>(socket, &size, 1);
  std::vector<T> res(size);
  receive_bytes<T>(socket, res.data(), size);
  return res;
}

template <typename T>
void send_value(SocketType socket, const T& v) {
  send_bytes<T>(socket, &v, 1);
}

template <typename T>
T receive_value(SocketType socket) {
  T v;
  receive_bytes<T>(socket, &v, 1);
  return v;
}

}  // namespace tcputils
}  // namespace distributed
}  // namespace phi
