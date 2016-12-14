/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#ifndef PADDLE_DISABLE_RDMA
#include "sxi_sock.h"
#else
#define PROMPT_ERR() LOG(FATAL) << "Paddle is not compiled with rdma"
#endif
#include "paddle/utils/Logging.h"

#include <netinet/in.h>
struct sxi_sock;
struct sxi_socket;

#ifndef MAX_VEC_SIZE
// define default MAX_VEC_SIZE
#define MAX_VEC_SIZE (1UL << 16)
#endif

namespace paddle {
/// Namespace rdma is adaptors for sxi_sock.h. Make paddle not depend on it
/// when disable rdma support
namespace rdma {
inline int numCpus() {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_num_configured_cpus();
#else
  return 0;
#endif
}

inline sxi_socket* ssocket(int cpuId) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_ssocket(cpuId);
#else
  PROMPT_ERR();
#endif
}

inline int listen(sxi_socket* s) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_listen(s);
#else
  PROMPT_ERR();
#endif
}

inline int bind(sxi_socket* s, const char* str) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_bind(s, str);
#else
  PROMPT_ERR();
#endif
}

inline sxi_sock* accept(sxi_socket* s) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_accept(s);
#else
  PROMPT_ERR();
#endif
}

inline sockaddr_in* getSourceAddress(sxi_sock* sock) {
#ifndef PADDLE_DISABLE_RDMA
  return reinterpret_cast<sockaddr_in*>(&sock->sa);
#else
  PROMPT_ERR();
#endif
}

inline int close(sxi_socket* sock) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_socket_close(sock);
#else
  PROMPT_ERR();
#endif
}

inline int close(sxi_sock* sock) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_sock_close(sock);
#else
  PROMPT_ERR();
#endif
}

inline void init() {
#ifndef PADDLE_DISABLE_RDMA
  sxi_module_init();
#else
  PROMPT_ERR();
#endif
}

inline sxi_socket* csocket(int cpuId) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_csocket(cpuId);
#else
  PROMPT_ERR();
#endif
}

inline ssize_t read(sxi_sock* channel, void* data, size_t len) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_read(channel, data, len);
#else
  PROMPT_ERR();
#endif
}

inline ssize_t write(sxi_sock* channel, void* data, size_t len) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_write(channel, data, len);
#else
  PROMPT_ERR();
#endif
}

inline ssize_t readv(sxi_sock* channel, iovec* iov, int count) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_readv(channel, iov, count);
#else
  PROMPT_ERR();
#endif
}

inline ssize_t writev(sxi_sock* channel, iovec* iov, int count) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_writev(channel, iov, count);
#else
  PROMPT_ERR();
#endif
}

inline sxi_sock* connect(sxi_socket* socket, const char* url) {
#ifndef PADDLE_DISABLE_RDMA
  return sxi_connect(socket, url);
#else
  PROMPT_ERR();
#endif
}

}  //  namespace rdma
}  //  namespace paddle
