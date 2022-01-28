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

#ifndef PADDLE_FLUID_DISTRIBUTED_SOCKET_TCP_UTILS_H_
#define PADDLE_FLUID_DISTRIBUTED_SOCKET_TCP_UTILS_H_

#include <netdb.h>
#include <netinet/tcp.h>
#include <chrono>
#include <iostream>
#include <vector>

// Utility functions for TCP socket.
namespace paddle {
namespace distributed {
namespace tcputils {

constexpr int LISTENQ = 2048;
constexpr std::chrono::seconds kNoTimeOut = std::chrono::seconds::zero();

std::error_code getSocketError();
::addrinfo* getAddrInfo(const std::string host, const std::string service,
                        int ai_flags, int family);
void freeAddrInfo(::addrinfo*);
int tcpConnect(const std::string host, const std::string service, int family,
               std::chrono::seconds timeout);
int tcpListen(const std::string host, const std::string service, int family);
int tcpAccept(int sock);
void setSockOpt(int sock, int level, int optname, const char* value,
                int opt_len);

template <typename T>
void sendBytes(int sock, const T* buffer, size_t len);
template <typename T>
void recvBytes(int sock, T* buffer, size_t len);
template <typename T>
void sendVector(int sock, const std::vector<T>& v);
template <typename T>
std::vector<T> recvVector(int sock);
template <typename T>
void sendValue(int sock, const T& v);
template <typename T>
T recvValue(int sock);
void sendString(int sock, const std::string& s);
std::string recvString(int sock);

}  // namespace tcputils
}  // namespace distributed
}  // namespace paddle

#endif  // PADDLE_FLUID_DISTRIBUTED_SOCKET_TCP_UTILS_H_
