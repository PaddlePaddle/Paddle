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

#ifndef PADDLE_FLUID_DISTRIBUTED_STORE_TCP_UTILS_H_
#define PADDLE_FLUID_DISTRIBUTED_STORE_TCP_UTILS_H_

#include <netdb.h>
#include <netinet/tcp.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

// Utility functions for TCP socket.
namespace paddle {
namespace distributed {
namespace tcputils {

constexpr int LISTENQ = 2048;
constexpr std::chrono::milliseconds kNoTimeOut =
    std::chrono::milliseconds::zero();

std::error_code getSocketError();
::addrinfo* getAddrInfo(const std::string host, const std::string service,
                        int ai_flags, int family);
void freeAddrInfo(::addrinfo*);
int tcpConnect(const std::string host, const std::string service, int family,
               std::chrono::milliseconds timeout);
int tcpListen(const std::string host, const std::string service, int family);
int tcpAccept(int sock);
void setSockOpt(int sock, int level, int optname, const char* value,
                int opt_len);

// template <typename T>
// void sendBytes(int sock, const T* buffer, size_t len);
// template <typename T>
// void recvBytes(int sock, T* buffer, size_t len);
// template <typename T>
// void sendVector(int sock, const std::vector<T>& v);
// template <typename T>
// std::vector<T> recvVector(int sock);
// template <typename T>
// void sendValue(int sock, const T& v);
// template <typename T>
// T recvValue(int sock);
void sendString(int sock, const std::string& s);
std::string recvString(int sock);
//
template <typename T>
void sendBytes(int sock, const T* buffer, size_t len) {
  size_t to_send = len * sizeof(T);
  if (to_send == 0) {
    return;
  }
  auto bytes = reinterpret_cast<const char*>(buffer);

  while (to_send > 0) {
    auto byte_sent = ::send(sock, bytes, to_send, 0);
    VLOG(0) << "byte sent: " << byte_sent;
    PADDLE_ENFORCE_GT(byte_sent, 0, platform::errors::InvalidArgument(
                                        "TCP send error. Details: %s.",
                                        std::strerror(errno)));

    to_send -= byte_sent;
    bytes += byte_sent;
  }
}

template <typename T>
void recvBytes(int sock, T* buffer, size_t len) {
  size_t to_recv = len * sizeof(T);
  if (to_recv == 0) {
    return;
  }
  auto bytes = reinterpret_cast<char*>(buffer);

  while (to_recv > 0) {
    auto byte_recv = ::recv(sock, bytes, to_recv, 0);
    PADDLE_ENFORCE_GT(byte_recv, 0, platform::errors::InvalidArgument(
                                        "TCP recv error. Details: %s.",
                                        std::strerror(errno)));

    to_recv -= byte_recv;
    bytes += byte_recv;
  }
}

template <typename T>
void sendValue(int sock, const T& v) {
  sendBytes<T>(sock, &v, 1);
}

template <typename T>
T recvValue(int sock) {
  T v;
  recvBytes<T>(sock, &v, 1);
  return v;
}

template <typename T>
void sendVector(int sock, const std::vector<T>& v) {
  size_t size = v.size();
  sendBytes<size_t>(sock, &size, 1);
  sendBytes<T>(sock, v.data(), size);
}

template <typename T>
std::vector<T> recvVector(int sock) {
  size_t size;
  recvBytes<size_t>(sock, &size, 1);
  std::vector<T> res(size);
  recvBytes<T>(sock, res.data(), size);
  return res;
}

}  // namespace tcputils
}  // namespace distributed
}  // namespace paddle

#endif  // PADDLE_FLUID_DISTRIBUTED_STORE_TCP_UTILS_H_
