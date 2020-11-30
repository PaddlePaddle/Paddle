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
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);

  int server_fd, new_socket;
  struct sockaddr_in address;
  int addrlen = sizeof(address);
  char buffer[1024] = {0};
  int opt = 0;
  // creating socket fd
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    PADDLE_THROW(
        platform::errors::Unavailable("Create server file descriptor failed."));
  }

  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    PADDLE_THROW(platform::errors::Unavailable("Set socket options failed."));
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  int try_times = 0;
  int retry_time = 0;
  while (true) {
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
      retry_time = 3 * (try_times + 1);
      LOG(WARNING) << "Socket bind worker " << ep
                   << (try_times < 9
                           ? " failed, try again after " +
                                 std::to_string(retry_time) + " seconds."
                           : " failed, try again after " +
                                 std::to_string(retry_time) +
                                 " seconds. Bind on endpoint " + ep +
                                 " failed. Please confirm whether the "
                                 "communication port or GPU card is occupied.");
      std::this_thread::sleep_for(std::chrono::seconds(retry_time));
      ++try_times;
      continue;
    }
    break;
  }

  VLOG(3) << "listening on: " << ep;
  if (listen(server_fd, 3) < 0) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Listen on server file descriptor failed."));
  }

  if ((new_socket =
           accept(server_fd, reinterpret_cast<struct sockaddr *>(&address),
                  reinterpret_cast<socklen_t *>(&addrlen))) < 0) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Accept the new socket file descriptor failed."));
  }

  if (read(new_socket, buffer, 1024) < 0) {
    PADDLE_THROW(platform::errors::Unavailable("Read from socket failed."));
  }

  VLOG(3) << "recevived the ncclUniqueId";
  memcpy(nccl_id, buffer, NCCL_UNIQUE_ID_BYTES);

  VLOG(3) << "closing the socket server: " << ep;
  close(server_fd);
}

void NCCLParallelContext::SendNCCLID(const std::string &ep,
                                     ncclUniqueId *nccl_id) {
  auto addr = paddle::string::Split(ep, ':');
  PADDLE_ENFORCE_EQ(
      addr.size(), 2UL,
      platform::errors::InvalidArgument(
          "The endpoint should contain host and port, but got %s.", ep));
  std::string host = addr[0];
  int port = std::stoi(addr[1]);
  // struct sockaddr_in address;
  int sock = 0;
  struct sockaddr_in serv_addr;
  char buffer[1024] = {0};

  memcpy(buffer, nccl_id, NCCL_UNIQUE_ID_BYTES);
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    PADDLE_THROW(platform::errors::Unavailable("Create socket failed."));
  }

  memset(&serv_addr, '0', sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  char *ip = NULL;
  struct hostent *hp;
  if ((hp = gethostbyname(host.c_str())) == NULL) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Fail to get host by name %s.", host));
  }
  int i = 0;
  while (hp->h_addr_list[i] != NULL) {
    ip = inet_ntoa(*(struct in_addr *)hp->h_addr_list[i]);
    VLOG(3) << "gethostbyname  host:" << host << "  ->ip: " << ip;
    break;
  }
  if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
    PADDLE_THROW(platform::errors::Unavailable("Open address %s failed.", ep));
  }

  int try_times = 0;
  int retry_time = 0;
  while (true) {
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      retry_time = 3 * (try_times + 1);
      LOG(WARNING)
          << "Socket connect worker " << ep
          << (try_times < 9
                  ? " failed, try again after " + std::to_string(retry_time) +
                        " seconds."
                  : " failed, try again after " + std::to_string(retry_time) +
                        " seconds. Maybe that some process is occupied the "
                        "GPUs of this node now, and you should kill those "
                        "process manually.");
      std::this_thread::sleep_for(std::chrono::seconds(retry_time));
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
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    ncclUniqueId nccl_id;
    if (strategy_.local_rank_ == 0) {
      // generate the unique ncclid on the root worker
      platform::dynload::ncclGetUniqueId(&nccl_id);
      BcastNCCLId(&nccl_id, 0);
    } else {
      BcastNCCLId(&nccl_id, 0);
    }
    int gpu_id = BOOST_GET_CONST(platform::CUDAPlace, place_).device;
    VLOG(0) << "init nccl context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " gpu id: " << gpu_id
            << " ring id: " << ring_id;

    // it will assign nccl_comm in CUDADeviceContext within ring_id
    platform::NCCLCommContext::Instance().CreateNCCLComm(
        &nccl_id, strategy_.nranks_, strategy_.local_rank_, gpu_id, ring_id);
  }
}

void NCCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place_), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place_);
  cudaStream_t stream = nullptr;
  if (use_calc_stream) {
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place_);
    stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
  } else {
    stream = comm->stream();
  }
  AllReduce(src, dst, strategy_, stream);
}

paddle::platform::CUDADeviceContext *NCCLParallelContext::GetDeviceContext(
    int ring_id) {
  return platform::NCCLCommContext::Instance()
      .Get(ring_id, place_)
      ->dev_context();
}

#endif

}  //  namespace imperative
}  //  namespace paddle
