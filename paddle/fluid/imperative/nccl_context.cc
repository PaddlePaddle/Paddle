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
  while (true) {
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
      LOG(WARNING) << "Socket bind worker " << ep
                   << (try_times < 5 ? " failed, try again after 3 seconds."
                                     : " failed, try again after 3 seconds. "
                                       "Bind on endpoint %s failed. "
                                       "Please confirm whether the "
                                       "communication port or GPU card is "
                                       "occupied.");
      std::this_thread::sleep_for(std::chrono::seconds(3));
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
  while (true) {
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      LOG(WARNING)
          << "Socket connect worker " << ep
          << (try_times < 5
                  ? " failed, try again after 3 seconds."
                  : " failed, try again after 3 seconds. Maybe that "
                    "some process is occupied the GPUs of this node "
                    "now, and you should kill those process manually.");
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

void NCCLParallelContext::AllReduce(const framework::Tensor &src,
                                    framework::Tensor *dst,
                                    paddle::platform::NCCLComm *comm,
                                    cudaStream_t stream) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));
  const void *src_ptr = src.data<void>();
  dst->Resize(src.dims());
  auto *dst_ptr = dst->mutable_data(src.place(), src.type());
  auto nccl_dtype = platform::ToNCCLDataType(src.type());
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
      src_ptr, dst_ptr, src.numel(), nccl_dtype, ncclSum, comm->comm(),
      stream));
}

#if NCCL_VERSION_CODE >= 2212
void NCCLParallelContext::AllReduce(const framework::SelectedRows &src,
                                    framework::SelectedRows *dst,
                                    const ParallelStrategy &strategy,
                                    cudaStream_t stream,
                                    paddle::platform::NCCLComm *comm) {
  VLOG(3) << "SelectedRows AllReduce start";
  const auto &src_tensor = src.value();
  const auto &place = src_tensor.place();
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));

  auto dtype = src_tensor.type();
  auto nccl_dtype = platform::ToNCCLDataType(dtype);

  // 1. Gather rows number from all workers. Here use ncclAllGather to do this,
  // but we can use other ways to implement is in the future
  const auto &src_rows = src.rows();
  framework::Vector<int64_t> rows_num_vector(strategy.nranks_);
  rows_num_vector[strategy.local_rank_] = static_cast<int64_t>(src_rows.size());
  auto *gpu_rows_num_ptr = rows_num_vector.CUDAMutableData(place);
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
      gpu_rows_num_ptr + strategy.local_rank_, gpu_rows_num_ptr, 1, ncclInt64,
      comm->comm(), stream));

  const auto *cpu_rows_num_ptr = rows_num_vector.data();
  auto rows_num =
      std::accumulate(cpu_rows_num_ptr, cpu_rows_num_ptr + strategy.nranks_,
                      static_cast<int64_t>(0));
  dst->set_height(src.height());

  VLOG(3) << "Gather rows: " << string::join_strings(rows_num_vector, ',')
          << ", total rows number: " << rows_num
          << ", height: " << src.height();

  auto *dst_rows = dst->mutable_rows();
  dst_rows->resize(rows_num);
  auto *dst_rows_ptr = dst_rows->CUDAMutableData(place);
  const auto *src_rows_ptr = src_rows.CUDAData(place);

  auto *dst_tensor = dst->mutable_value();
  auto dims = src_tensor.dims();
  dims[0] = rows_num;
  auto feature_size = framework::product(dims) / dims[0];
  dst_tensor->Resize(dims);
  auto *dst_tensor_ptr = dst_tensor->mutable_data(place, dtype);
  const auto *src_tensor_ptr = src_tensor.data<void>();

  auto sizeof_dtype = framework::SizeOfType(dtype);
  int64_t row_offset = 0;
  for (int i = 0; i < strategy.nranks_; ++i) {
    if (cpu_rows_num_ptr[i] > 0) {
      // 2. Broadcast the rows of SelectedRows
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBroadcast(
          src_rows_ptr, dst_rows_ptr + row_offset, cpu_rows_num_ptr[i],
          ncclInt64, i, comm->comm(), stream));
      // 3. Broadcast the tensor data of SelectedRows
      auto *dst_tensor_ptr_i = reinterpret_cast<uint8_t *>(dst_tensor_ptr) +
                               row_offset * feature_size * sizeof_dtype;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBroadcast(
          src_tensor_ptr, dst_tensor_ptr_i, cpu_rows_num_ptr[i] * feature_size,
          nccl_dtype, i, comm->comm(), stream));
      row_offset += cpu_rows_num_ptr[i];
    }
  }

  VLOG(3) << "Original SelectedRows rows: "
          << string::join_strings(src_rows, ',');
  VLOG(3) << "Result SelectedRows rows: "
          << string::join_strings(*dst_rows, ',');
}
#endif

const platform::Place &NCCLParallelContext::GetVarPlace(
    const framework::Variable &src) {
  if (src.IsType<framework::LoDTensor>()) {
    return src.Get<framework::LoDTensor>().place();
#if NCCL_VERSION_CODE >= 2212
  } else if (src.IsType<framework::SelectedRows>()) {
    return src.Get<framework::SelectedRows>().value().place();
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Cannot get unsupported variable type %s for imperative allreduce, "
        "only "
        "LoDTensor and SelectedRows are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void NCCLParallelContext::AllReduce(const framework::Variable &src,
                                    framework::Variable *dst, int ring_id,
                                    bool use_calc_stream) {
  const auto &place = GetVarPlace(src);
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
  cudaStream_t stream = nullptr;
  if (use_calc_stream) {
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
  } else {
    stream = comm->stream();
  }

  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(), comm, stream);
#if NCCL_VERSION_CODE >= 2212
  } else if (src.IsType<framework::SelectedRows>()) {
    if (&src != dst) {
      if (!dst->IsType<framework::SelectedRows>()) {
        dst->Clear();
      }
      AllReduce(src.Get<framework::SelectedRows>(),
                dst->GetMutable<framework::SelectedRows>(), strategy_, stream,
                comm);
    } else {
      // SelectedRows cannot be allreduce in-place
      framework::Variable tmp_dst;
      AllReduce(src.Get<framework::SelectedRows>(),
                tmp_dst.GetMutable<framework::SelectedRows>(), strategy_,
                stream, comm);
      *dst = std::move(tmp_dst);
    }
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "LoDTensor and SelectedRows are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void NCCLParallelContext::SyncCalcStream(const platform::Place &place) {
  auto dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(dev_ctx->stream()));
}

void NCCLParallelContext::SyncCommStream(const platform::Place &place,
                                         int ring_id) {
  auto stream =
      platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
}
#endif

}  //  namespace imperative
}  //  namespace paddle
