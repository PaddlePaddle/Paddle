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

#include "paddle/phi/core/distributed/store/socket.h"
#include <array>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <cstdio>

namespace phi::distributed {

#ifdef _WIN32
static int _get_sockname_of_win(int sock, char* out, int out_len) {
  snprintf(out, out_len, "not support win now");
  return 0;
}
#else
static int _get_sockname(int sock, char *out, int out_len) {
  struct sockaddr_in addr = {};
  socklen_t s_len = sizeof(addr);

  if (::getpeername(sock, reinterpret_cast<sockaddr *>(&addr), &s_len)) {
    ::snprintf(
        out, out_len, "can't getsocketname of %d, errno:%d", sock, errno);
    return -1;
  }

  std::array<char, 128> ip = {};
  int port = 0;

  // deal with both IPv4 and IPv6:
  if (addr.sin_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&addr;
    port = ntohs(s->sin_port);
    ::inet_ntop(AF_INET, &s->sin_addr, ip.data(), sizeof(ip));
  } else {  // AF_INET6
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
    port = ntohs(s->sin6_port);
    ::inet_ntop(AF_INET6, &s->sin6_addr, ip.data(), sizeof(ip));
  }

  ::snprintf(out, out_len, "%s:%d", ip.data(), port);
  return 0;
}
#endif

int GetSockName(int sock, char* out, int out_len) {
#ifdef _WIN32
  return _get_sockname_of_win(sock, out, out_len);
#else
  return _get_sockname(sock, out, out_len);
#endif
}

std::string GetSockName(int fd) {
  std::array<char, 256> out = {};
  GetSockName(fd, out.data(), sizeof(out));
  return std::string(out.data());
}

}  // namespace phi::distributed
