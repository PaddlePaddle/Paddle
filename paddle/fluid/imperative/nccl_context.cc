//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/nccl_context.h"

namespace paddle {
namespace imperative {
#if defined(PADDLE_WITH_NCCL)
void NCCLParallelContext::RecvNCCLID(const std::string &ep,
                                     ncclUniqueId *nccl_id) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(addr.size(), 2UL,
                    "The endpoint should contain host and port: %s", ep);
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  int server_fd, new_socket;
  struct sockaddr_in address;
  int addrlen = sizeof(address);
  char buffer[1024] = {0};
  int opt = 0;
  // creating socket fd
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    PADDLE_THROW("create server fd failed");
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    PADDLE_THROW("set socket opt failed");

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    PADDLE_THROW("binding failed on ep: %s", ep);
  VLOG(3) << "listening on: " << ep;
  if (listen(server_fd, 3) < 0) PADDLE_THROW("listen on server fd failed");

  if ((new_socket =
           accept(server_fd, reinterpret_cast<struct sockaddr *>(&address),
                  reinterpret_cast<socklen_t *>(&addrlen))) < 0)
    PADDLE_THROW("accept the new socket fd failed");

  if (read(new_socket, buffer, 1024) < 0)
    PADDLE_THROW("reading the ncclUniqueId from socket failed");
  VLOG(3) << "recevived the ncclUniqueId";
  memcpy(nccl_id, buffer, NCCL_UNIQUE_ID_BYTES);

  VLOG(3) << "closing the socket server: " << ep;
  close(server_fd);
}

void NCCLParallelContext::SendNCCLID(const std::string &ep,
                                     ncclUniqueId *nccl_id) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(addr.size(), 2UL,
                    "The endpoint should contain host and port: %s", ep);
  std::string host = addr[0];
  int port = std::stoi(addr[1]);
  // struct sockaddr_in address;
  int sock = 0;
  struct sockaddr_in serv_addr;
  char buffer[1024] = {0};

  memcpy(buffer, nccl_id, NCCL_UNIQUE_ID_BYTES);
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    PADDLE_THROW("create socket failed");

  memset(&serv_addr, '0', sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0)
    PADDLE_THROW("invalied address: %s", ep);

  int try_times = 0;
  while (true) {
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      VLOG(0) << "worker: " << ep
              << (try_times < 5 ? " is not ready, will retry after 3 seconds..."
                                : " is not ready. Maybe that some process "
                                  "is occupied the GPUs of this node now, "
                                  "and you should kill those process manually. "
                                  "Will retry after 3 seconds...");

      std::this_thread::sleep_for(std::chrono::seconds(3));
      ++try_times;
      continue;
    }
    VLOG(3) << "sending the ncclUniqueId to " << ep;
    send(sock, buffer, NCCL_UNIQUE_ID_BYTES, 0);
    break;
  }
  close(sock);
}

void NCCLParallelContext::BcastNCCLId(ncclUniqueId *nccl_id, int root) {
  if (strategy_.local_rank_ == root) {
    for (auto ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) SendNCCLID(ep, nccl_id);
    }
  } else {
    RecvNCCLID(strategy_.current_endpoint_, nccl_id);
  }
}

void NCCLParallelContext::Init() {
  ncclUniqueId nccl_id;
  ncclComm_t comm;
  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    platform::dynload::ncclGetUniqueId(&nccl_id);
    BcastNCCLId(&nccl_id, 0);
  } else {
    BcastNCCLId(&nccl_id, 0);
  }
  int gpu_id = boost::get<platform::CUDAPlace>(place_).device;
  VLOG(0) << "init nccl context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " gpu id: " << gpu_id;

  PADDLE_ENFORCE(cudaSetDevice(gpu_id));
  PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
      &comm, strategy_.nranks_, nccl_id, strategy_.local_rank_));

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(pool.Get(place_));
  dev_ctx->set_nccl_comm(comm);
}
#endif

}  //  namespace imperative
}  //  namespace paddle
