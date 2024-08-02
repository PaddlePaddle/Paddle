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
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/service/simple_rpc/rpc_server.h"
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include "paddle/fluid/distributed/ps/service/simple_rpc/baidu_rpc_server.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {
namespace simple {
RpcService::RpcService(RpcCallback callback) : _callback(std::move(callback)) {
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  void* my_ptr = reinterpret_cast<void*>(this);
  std::vector<void*> ids = gloo->AllGather(my_ptr);
  _remote_ptrs.assign(gloo->Size(), NULL);
  for (int i = 0; i < gloo->Size(); ++i) {
    _remote_ptrs[i] = reinterpret_cast<RpcService*>(ids[i]);
  }
  gloo->Barrier();
}
RpcService::~RpcService() {
  paddle::framework::GlooWrapper::GetInstance()->Barrier();
  if (_request_counter != 0) {
    fprintf(stderr, "check request counter is not zero");
  }
}

inline uint32_t get_broadcast_ip(char* ethname) {
  struct ifreq ifr;
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  strncpy(ifr.ifr_name, ethname, IFNAMSIZ - 1);
  if (ioctl(sockfd, SIOCGIFBRDADDR, &ifr) == -1) {
    return 0;
  }
  close(sockfd);
  return ((struct sockaddr_in*)&ifr.ifr_addr)->sin_addr.s_addr;
}
inline std::string get_local_ip_internal() {
  int sockfd = -1;
  char buf[512];
  struct ifconf ifconf;
  struct ifreq* ifreq;

  ifconf.ifc_len = 512;
  ifconf.ifc_buf = buf;
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  PADDLE_ENFORCE_EQ(
      (sockfd >= 0),
      true,
      common::errors::PreconditionNotMet("Socket should be >= 0."));
  int ret = ioctl(sockfd, SIOCGIFCONF, &ifconf);
  PADDLE_ENFORCE_EQ(
      (ret >= 0),
      true,
      common::errors::PreconditionNotMet("Ioctl ret should be >= 0."));
  ret = close(sockfd);
  PADDLE_ENFORCE_EQ(
      (0 == ret),
      true,
      common::errors::PreconditionNotMet("Close call should return 0."));

  ifreq = (struct ifreq*)buf;
  for (int i = 0; i < static_cast<int>(ifconf.ifc_len / sizeof(struct ifreq));
       i++) {
    std::string ip =
        inet_ntoa(((struct sockaddr_in*)&ifreq->ifr_addr)->sin_addr);
    if (strncmp(ifreq->ifr_name, "lo", 2) == 0 ||
        strncmp(ifreq->ifr_name, "docker", 6) == 0) {
      fprintf(stdout,
              "skip interface: [%s], ip: %s\n",
              ifreq->ifr_name,
              ip.c_str());
      ifreq++;
      continue;
    }
    if (get_broadcast_ip(ifreq->ifr_name) == 0) {
      fprintf(stdout,
              "skip interface: [%s], ip: %s\n",
              ifreq->ifr_name,
              ip.c_str());
      ifreq++;
      continue;
    }
    if (ip != "127.0.0.1") {
      fprintf(stdout,
              "used interface: [%s], ip: %s\n",
              ifreq->ifr_name,
              ip.c_str());
      return ip;
    }
    ifreq++;
  }
  fprintf(stdout, "not found, use ip: 127.0.0.1\n");
  return "127.0.0.1";
}
RpcServer::RpcServer() {
  _gloo = paddle::framework::GlooWrapper::GetInstance().get();
  std::string ip = get_local_ip_internal();
  uint32_t int_ip = inet_addr(ip.c_str());
  _ips = _gloo->AllGather(int_ip);
}
RpcServer::~RpcServer() {
  if (_gloo != NULL) {
    _gloo = NULL;
  }
}
void RpcServer::set_connection_num(int n) {
  _gloo->Barrier();
  if (n < _gloo->Size()) {
    n = _gloo->Size();
  }
  PADDLE_ENFORCE_EQ(
      (n >= 1),
      true,
      common::errors::InvalidArgument("Connect num need more than 1."));
  _conn_num = n;
}
void RpcServer::set_thread_num(int n) {
  if (n < _gloo->Size()) {
    n = _gloo->Size();
  }
  PADDLE_ENFORCE_EQ(
      (n >= 1),
      true,
      common::errors::InvalidArgument("Thread num need more than 1."));
  _thread_num = n;
}
void* RpcServer::add_service(RpcCallback callback, bool simplex) {
  return new RpcService(std::move(callback));
}
void RpcServer::remove_service(void* service) {
  delete reinterpret_cast<RpcService*>(service);
}
RpcServer& global_rpc_server() {
  static BaiduRpcServer server;
  return server;
}
}  // namespace simple
}  // namespace distributed
}  // namespace paddle
#endif
