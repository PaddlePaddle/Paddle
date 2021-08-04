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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/gen_comm_id_helper.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <algorithm>
#include <string>
#include <thread>  // NOLINT

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace paddle {
namespace platform {

std::once_flag SocketServer::init_flag_;

struct CommHead {
  int version = 1;  // unused for now
  int ring_id = 0;
};

// Check system calls, such as socket, bind.
#define CHECK_SYS_CALL(call, name)          \
  do {                                      \
    int retval;                             \
    CHECK_SYS_CALL_VAL(call, name, retval); \
  } while (false)

#define CHECK_SYS_CALL_VAL(call, name, retval)                            \
  do {                                                                    \
    RETRY_SYS_CALL_VAL(call, name, retval);                               \
    if (retval == -1) {                                                   \
      PADDLE_THROW(platform::errors::Unavailable("Call to %s failed: %s", \
                                                 name, strerror(errno))); \
    }                                                                     \
  } while (false)

#define RETRY_SYS_CALL_VAL(call, name, retval)                           \
  do {                                                                   \
    retval = (call);                                                     \
    if (retval == -1 &&                                                  \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {   \
      LOG(WARNING) << "Call " << name << " returned " << strerror(errno) \
                   << " retry";                                          \
    } else {                                                             \
      break;                                                             \
    }                                                                    \
  } while (true)

static int SocketSend(int fd, const char* buffer, int size) {
  int offset = 0;
  int bytes = 0;
  while (offset < size) {
    bytes = send(fd, buffer + offset, size - offset, 0);
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        // send failed
        return -1;
      } else {
        bytes = 0;
      }
    }
    offset += bytes;
  }
  return offset;
}

static int SocketRecv(int fd, char* buffer, int size) {
  int offset = 0;
  int bytes = 0;
  while (offset < size) {
    bytes = recv(fd, buffer + offset, size - offset, 0);
    if (bytes == 0) {
      // closed by client, maybe probing alive client
      return 0;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        return -1;
      } else {
        bytes = 0;
      }
    }
    offset += bytes;
  }
  return offset;
}

static void BindOrConnectFailed(int timeout, int* try_times, int* total_time,
                                const char* op, const std::string& ep) {
  PADDLE_ENFORCE_LT(
      *total_time, timeout,
      platform::errors::Unavailable("%s addr=%s timeout, failed reason: %s", op,
                                    ep.c_str(), strerror(errno)));
  ++(*try_times);
  int retry_time = std::min(*try_times * 500, 3000);  // max 3 seconds
  *total_time += retry_time;

  LOG(WARNING) << op << " addr=" << ep << " failed " << *try_times
               << " times with reason: " << strerror(errno) << " retry after "
               << retry_time / 1000.0 << " seconds";
  std::this_thread::sleep_for(std::chrono::milliseconds(retry_time));
}

int CreateListenSocket(const std::string& ep) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  // creating socket fd
  int server_fd = -1;
  CHECK_SYS_CALL_VAL(socket(AF_INET, SOCK_STREAM, 0), "socket", server_fd);

  // NOTE. Solutions to `Address already in use`.
  // 1. Reuse addr&port. Otherwise, once the server closes the socket
  // before client, the server will enter TIME-WAIT status. If we bind port
  // again, the error `Address already in use` will appear.
  // 2. Or we can close the client first to ensure that the server does
  // not enter the TIME-WAIT state. But this is obviously not as convenient
  // as the reuse method.
  int opt = 1;
#if defined(SO_REUSEPORT)
  // since Linux kernel 3.9
  CHECK_SYS_CALL(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                            &opt, sizeof(opt)),
                 "setsockopt");
#else
  CHECK_SYS_CALL(
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
      "setsockopt");
#endif

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  // TODO(wangxi) Set from env, default 900s=15min
  int timeout = 900 * 1000;
  int try_times = 0;
  int total_time = 0;
  while (true) {
    int ret_val = -1;
    RETRY_SYS_CALL_VAL(
        bind(server_fd, (struct sockaddr*)&address, sizeof(address)), "bind",
        ret_val);

    if (ret_val == -1) {
      BindOrConnectFailed(timeout, &try_times, &total_time, "bind", ep);
      continue;
    }
    break;
  }

  CHECK_SYS_CALL(listen(server_fd, 3), "listen");
  LOG(INFO) << "Server listening on: " << ep << " successful.";
  return server_fd;
}

void CloseSocket(int fd) { CHECK_SYS_CALL(close(fd), "close"); }

static int SocketAccept(int server_fd, const CommHead head) {
  static_assert(sizeof(CommHead) <= 1024,
                "sizeof(CommHead) must <= buffer size");

  struct sockaddr_in client_addr;
  socklen_t addr_length = sizeof(client_addr);
  char buffer[1024] = {0};
  int conn = -1;
  const char* phead = reinterpret_cast<const char*>(&head);

  while (true) {
    CHECK_SYS_CALL_VAL(
        accept(server_fd, reinterpret_cast<struct sockaddr*>(&client_addr),
               &addr_length),
        "accept", conn);

    int ret_val = SocketRecv(conn, buffer, sizeof(head));
    if (ret_val > 0 && memcmp(buffer, phead, sizeof(head)) == 0) {
      // send a message to the sender, indicating that the link is correct
      CHECK_SYS_CALL(SocketSend(conn, phead, sizeof(head)), "send");
      break;  // accept client
    } else {
      VLOG(3) << "socket read failed with ret_val=" << ret_val;
      CloseSocket(conn);
    }
  }
  return conn;
}

static int ConnectAddr(const std::string& ep, const CommHead head) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);

  char* ip = NULL;
  struct hostent* hp = NULL;
  hp = gethostbyname(host.c_str());
  PADDLE_ENFORCE_NOT_NULL(hp, platform::errors::InvalidArgument(
                                  "Fail to get host by name %s.", host));

  int i = 0;
  while (hp->h_addr_list[i] != NULL) {
    ip = inet_ntoa(*(struct in_addr*)hp->h_addr_list[i]);
    VLOG(3) << "gethostbyname  host:" << host << "  ->ip: " << ip;
    break;
  }

  PADDLE_ENFORCE_GT(inet_pton(AF_INET, ip, &server_addr.sin_addr), 0,
                    platform::errors::Unavailable("Open address %s failed: %s",
                                                  ep, strerror(errno)));

  static_assert(sizeof(CommHead) <= 1024,
                "sizeof(CommHead) must <= buffer size");
  char buffer[1024] = {0};
  const char* phead = reinterpret_cast<const char*>(&head);

  // TODO(wangxi) Set from env, default 900s=15min
  int timeout = 900 * 1000;
  int try_times = 0;
  int total_time = 0;

  int sock = -1;
  CHECK_SYS_CALL_VAL(socket(AF_INET, SOCK_STREAM, 0), "socket", sock);
  while (true) {
    int ret_val = -1;
    RETRY_SYS_CALL_VAL(
        connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)),
        "connect", ret_val);

    if (ret_val == -1) {
      BindOrConnectFailed(timeout, &try_times, &total_time, "connect", ep);
      continue;
    }

    CHECK_SYS_CALL(SocketSend(sock, phead, sizeof(head)), "send");
    ret_val = SocketRecv(sock, buffer, sizeof(head));
    if (ret_val > 0 && memcmp(buffer, phead, sizeof(head)) == 0) {
      // recv same message from recver, indicating that the link is correct
      break;  // accept client
    } else {
      VLOG(3) << "socket read failed with ret_val=" << ret_val;
      CloseSocket(sock);
    }
    sock = -1;
    CHECK_SYS_CALL_VAL(socket(AF_INET, SOCK_STREAM, 0), "socket", sock);
    // unmatched link, retry after 80ms
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
  }
  return sock;
}

// TODO(WANGXI): maybe need to unify this hard code
#ifdef PADDLE_WITH_ASCEND_CL
#define MAX_COMMUNIQUEID_LEN 4108
#else
#define MAX_COMMUNIQUEID_LEN 1024
#endif

template <typename CommUniqueId>
static void RecvCommID(int conn, CommUniqueId* nccl_id) {
  char buffer[MAX_COMMUNIQUEID_LEN] = {0};
  static_assert(sizeof(CommUniqueId) <= MAX_COMMUNIQUEID_LEN,
                "nccl id bytes must <= buffer size");

  CHECK_SYS_CALL(SocketRecv(conn, buffer, sizeof(CommUniqueId)),
                 "recv comm unique id");
  memcpy(nccl_id, buffer, sizeof(CommUniqueId));
}

template <typename CommUniqueId>
static void SendCommID(int conn, CommUniqueId* nccl_id) {
  char buffer[MAX_COMMUNIQUEID_LEN] = {0};
  memcpy(buffer, nccl_id, sizeof(CommUniqueId));

  CHECK_SYS_CALL(SocketSend(conn, buffer, sizeof(CommUniqueId)),
                 "send comm unique id");
}

template <typename CommUniqueId>
void SendBroadCastCommID(std::vector<std::string> servers,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id) {
  CommHead head;
  head.ring_id = ring_id;

  // connect with server
  std::vector<int> connects;
  for (auto server : servers) {
    VLOG(3) << "connecting endpoint: " << server;
    int conn = ConnectAddr(server, head);
    connects.push_back(conn);
  }
  VLOG(3) << "connecting completed...";

  for (size_t i = 0; i < nccl_ids->size(); ++i) {
    int j = 0;
    for (auto conn : connects) {
      VLOG(3) << "sending comm_id to " << servers[j] << " nccl_comm_no: " << i;
      SendCommID(conn, &(*nccl_ids)[i]);
      ++j;
    }
  }

  // close client
  for (auto conn : connects) {
    CloseSocket(conn);
  }
}

template <typename CommUniqueId>
void RecvBroadCastCommID(std::string endpoint,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id) {
  int server = CreateListenSocket(endpoint);
  RecvBroadCastCommID(server, endpoint, nccl_ids, ring_id);
  CloseSocket(server);
}

template <typename CommUniqueId>
void RecvBroadCastCommID(int server_fd, std::string endpoint,
                         std::vector<CommUniqueId>* nccl_ids, int ring_id) {
  CommHead head;
  head.ring_id = ring_id;
  int client = SocketAccept(server_fd, head);

  for (size_t i = 0; i < nccl_ids->size(); ++i) {
    VLOG(3) << "trainer: " << endpoint
            << " receiving comm_id from trainer 0, nccl_comm_no: " << i;
    RecvCommID(client, &(*nccl_ids)[i]);
  }

  VLOG(3) << "receiving completed...";
  CloseSocket(client);
}

SocketServer& SocketServer::GetInstance(const std::string& end_point) {
  static SocketServer instance;
  std::call_once(init_flag_, [&]() {
    instance.server_fd_ = CreateListenSocket(end_point);
    instance.end_point_ = end_point;
  });
  PADDLE_ENFORCE_NE(instance.server_fd_, -1,
                    platform::errors::Unavailable(
                        "listen socket failed with end_point=%s", end_point));
  PADDLE_ENFORCE_EQ(instance.end_point_, end_point,
                    platform::errors::InvalidArgument(
                        "old end_point=%s must equal with new end_point=%s",
                        instance.end_point_, end_point));
  return instance;
}

/// template instantiation
#define INSTANT_TEMPLATE(Type)                                                 \
  template void SendBroadCastCommID<Type>(std::vector<std::string> servers,    \
                                          std::vector<Type> * nccl_ids,        \
                                          int ring_id = 0);                    \
  template void RecvBroadCastCommID<Type>(                                     \
      std::string endpoint, std::vector<Type> * nccl_ids, int ring_id = 0);    \
  template void RecvBroadCastCommID<Type>(int server_fd, std::string endpoint, \
                                          std::vector<Type>* nccl_ids,         \
                                          int ring_id = 0);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
INSTANT_TEMPLATE(ncclUniqueId)
#endif
#ifdef PADDLE_WITH_XPU_BKCL
INSTANT_TEMPLATE(BKCLUniqueId)
#endif
#ifdef PADDLE_WITH_ASCEND_CL
INSTANT_TEMPLATE(HcclRootInfo)
#endif
}  // namespace platform
}  // namespace paddle

#endif
