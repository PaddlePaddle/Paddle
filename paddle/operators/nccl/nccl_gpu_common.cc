#include "paddle/operators/nccl/nccl_gpu_common.h"
#include "paddle/platform/gpu_info.h"

namespace paddle {
namespace platform {

NCCLManager::NCCLManager() {}

NCCLManager::~NCCLManager() {
  for (auto& p : comm_table) {
    auto* comm = p.second;
    auto& gpus_ = comm->gpus_;
    for (int i = 0; i < gpus_.size(); ++i) {
      int gid = gpus_[i];
      platform::SetDeviceId(gid);

      // mapping gid to idx
      int idx = gid % gpus_.size();
      // wait finish
      NCCL_CHECK(
          cudaStreamWaitEvent(*comm->streams_[idx], comm->events_[idx], 0));

      NCCL_CHECK(cudaEventDestroy(comm->events_[idx]));

      NCCL_CHECK(ncclCommDestroy(comm->comms_[idx]));
    }
    delete comm;
  }
}

Communicator* NCCLManager::GetCommunicator(const std::vector<int>& gpus) const {
  std::string key;
  for (auto& id : gpus) {
    key += std::to_string(id);
  }
  std::sort(key.begin(), key.end());

  std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  auto* comm = comm_table[key];
  if (comm == nullptr) {
    comm = new Communicator(gpus.size());
    NCCL_CHECK(ncclCommInitAll(comm->comms_.data(), gpus.size(), gpus.data()));

    for (size_t i = 0; i < gpus.size(); ++i) {
      platform::SetDeviceId(gpus[i]);

      // block wait
      NCCL_CHECK(cudaEventCreateWithFlags(
          &events_[i], cudaEventBlockingSync | cudaEventDisableTiming));
    }
    comm_table[key] = comm;
  }
  return comm;
}

}  // namespace operators
}  // namespace paddle
