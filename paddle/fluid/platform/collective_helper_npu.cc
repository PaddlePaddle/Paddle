//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <time.h>
#include <utility>

DECLARE_bool(avoid_hccl_port_conflict);

namespace paddle {
namespace platform {

class HCCLCommImpl : public HCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override {
    return BOOST_GET_CONST(NPUPlace, dev_ctx_->GetPlace()).device;
  }

  ~HCCLCommImpl() {
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclCommDestroy(comm_));
  }

  void set_comm(HcclComm comm) { comm_ = comm; }
  HcclComm comm() const override { return comm_; }

  aclrtStream stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<NPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  NPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  HcclComm comm_;
  std::unique_ptr<NPUDeviceContext> dev_ctx_;
};

HCCLComm* HCCLCommContext::CreateHCCLComm(HcclRootInfo* hccl_id, int nranks,
                                          int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(hccl_id,
                          platform::errors::InvalidArgument(
                              "The hccl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks, 1,
      platform::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank, nranks,
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank, nranks));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  HcclComm comm;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
  VLOG(1) << "initialized comm: " << &comm << ", nranks: " << nranks
          << ", hccl_id: " << hccl_id << ", rank: " << rank;
  PADDLE_ENFORCE_NPU_SUCCESS(
      platform::dynload::HcclCommInitRootInfo(nranks, hccl_id, rank, &comm));

  VLOG(1) << "initialized comm: " << &comm << ", nranks: " << nranks
          << ", hccl_id: " << hccl_id << ", rank: " << rank;

  auto* comm_wrapper = AssignHCCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "hccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id
          << ", with comm: " << comm_wrapper->comm();

  std::call_once(once_flag_, []() {
    std::atexit([]() { HCCLCommContext::Instance().ReleaseHCCLComms(); });
  });

  return comm_wrapper;
}

HCCLComm* HCCLCommContext::AssignHCCLComm(HcclComm comm, int nranks, int rank,
                                          int dev_id, int ring_id) {
  std::unique_ptr<NPUDeviceContext> dev_ctx(
      new NPUDeviceContext(NPUPlace(dev_id)));

  HCCLCommImpl* c = new HCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<HCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<HCCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::NPUPlace(dev_id)));
    dev_ctx->set_hccl_comm(comm);
  }

  return comm_map_[ring_id][dev_id].get();
}

void HCCLCommContext::ReleaseHCCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

static const int g_hccl_port_start = 60000;
static const int g_hccl_port_end = 60015;
static int g_avoid_hccl_ports_steps = 0;

int GetSocketPort(int fd) {
  struct sockaddr_in local;
  socklen_t size = sizeof(local);
  if (0 != getsockname(fd, (struct sockaddr*)&local, &size)) {  // NOLINT
    return -1;
  }

  auto port = htons(local.sin_port);
  return port;
}

void WaitPortClosed(const std::vector<int>& ports) {
  if (!FLAGS_avoid_hccl_port_conflict) {
    return;
  }

  VLOG(10) << "check local port";
  bool conflict = false;
  int port = 0;
  for (auto s : ports) {
    VLOG(10) << "use local port:" << s;
    if ((s >= 60000 && s <= 60015) || s < 0) {
      conflict = true;
      port = s;
    }
  }

  if (conflict) {
    LOG(INFO) << "find local conflict port so wait 2MSL time, port:" << port;
    std::this_thread::sleep_for(std::chrono::seconds(123));
  }
}

static int Bind(int port) {
  struct sockaddr_in my_addr;
  int client = socket(AF_INET, SOCK_STREAM, 0);
  PADDLE_ENFORCE_GT(client, 0, "socket must be created");

  // Explicitly assigning port number 12010 by
  // binding client with that port
  my_addr.sin_family = AF_INET;
  my_addr.sin_addr.s_addr = INADDR_ANY;
  my_addr.sin_port = htons(port);

  // This ip address will change according to the machine
  my_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
  int ret =
      bind(client, (struct sockaddr*)&my_addr, sizeof(struct sockaddr_in));
  if (ret != 0) {
    close(conn);
    return -1;
  }

  return client;
}

static int WaitToBind(int port) {
  while (1) {
    int ret = Bind(port);
    if (ret < 0) {
      LOG(WARNING) << "bind to addr error wait to bind port :" << port
                   << ",ret:" << ret;
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    VLOG(10) << "bind to port:" << port << " OK";
    break;
  }

  return client;
}

void WaitToBind(const std::vector<int>& ports) {
  std::vector<int> conns;
  for (auto port : ports) {
    int conn = WaitToBind(port);
    conns.push_back(conn);
  }

  for (auto conn : conns) {
    close(conn);
  }
}

void prepare_dir(const std::string& dir_name) {
  struct stat st;
  if (stat(dir_name.c_str(), &st) == -1) {
    mkdir(dir_name.c_str(), 0700);
  }
}

std::vector<HCCLConn_> TryToProtectHcclFreePorts() {
  std::vector<HCCLConn_> conns;
  for (int i = g_hccl_port_start; i <= g_hccl_port_end; i++) {
    struct HCCLConn_ conn;
    conn.port = -1;
    conn.socket = -1;
    conns.push_back(conn);
  }

  struct timeval start;
  struct timeval now;
  gettimeofday(&start, NULL);
  while (1) {
    bool all_ok = true;
    for (int i = 0; i < conns.size(); i++) {
      if (conn[i].socket <= 0) {
        continue;
      }

      int ret = Bind(port);
      if (ret < 0) {
        all_ok = false;
      }

      conn[i].socket = ret;
    }

    if (all_ok) {
      break;
    }

    gettimeofday(&now, NULL);
    int64_t elapsed = (now.tv_sec - start.tv_sec);
    if (elapsed >= 123) {  // < 2MSL
      break;
    }
  }

  std::ostringstream ss;
  ss << "find free ports, size:" << conns.size() << ", ports:";

  for (int i = 0; i < conns.size(); i++) {
    ss << conns[i].port << ","
  }

  LOG(INFO) << ss.str();

  for (int i = 0; i < conns.size(); i++) {
    close(conns[i].socket);
  }

  if (conns.size() == 0) {
    LOG(WARNING) << "not find hccl free ports";
  }

  return conns;
}

void WaitHcclPorts(int device_id) {
  prepare_dir(".flags");

  LOG(INFO) << "begin to avoid hccl conflicts steps:"
            << g_avoid_hccl_ports_steps << ", device_id" << device_id;

  if (device_id == 0) {
    TryToProtectHcclFreePorts();
    for (int i = 1; i < 8; i++) {
      std::string file = string::Sprintf(".flags/hccl_flags_%d_%d",
                                         g_avoid_hccl_ports_steps, i);
      std::string tmp = file + ".tmp";
      FILE* fp = fopen(tmp.c_str(), "wb");
      fclose(fp);
      VLOG(4) << "touch file " << tmp;
      int ret = rename(tmp.c_str(), file.c_str());
      VLOG(4) << "rename file from " << tmp << " to " << file;
      PADDLE_ENFORCE_EQ(ret, 0, platform::errors::Fatal(
                                    "rename from %s to %s error, retcode:%d",
                                    tmp, file, ret));
    }
  } else {
    std::string file = string::Sprintf(".flags/hccl_flags_%d_%d",
                                       g_avoid_hccl_ports_steps, device_id);
    struct stat buf;
    while (true) {
      if (stat(file.c_str(), &buf) == 0) {
        VLOG(4) << "remove file " << file;
        remove(file.c_str());
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  LOG(INFO) << "begin to 2MSL time under steps:" << g_avoid_hccl_ports_steps
            << ", device_id" << device_id;
  std::this_thread::sleep_for(std::chrono::seconds(120));
  g_avoid_hccl_ports_steps += 1;
  LOG(INFO) << "end to wait to avoid hccl conflicts steps:"
            << g_avoid_hccl_ports_steps << ", device_id" << device_id;
}

}  // namespace platform
}  // namespace paddle
#endif
