#include "paddle/operators/nccl/nccl_gpu_common.h"
#include "paddle/platform/gpu_info.h"

namespace paddle {
namespace platform {

NCCLManager::NCCLManager() {}

NCCLManager::~NCCLManager() {
  for (auto& p : comm_table) {
    auto& comm = p.second;
    auto& gpus_ = comm->gpus_;
    for (size_t i = 0; i < gpus_.size(); ++i) {
      int gid = gpus_[i];
      platform::SetDeviceId(gid);

      // mapping gid to idx
      int idx = gid % gpus_.size();
      // wait finish
      PADDLE_ENFORCE(
          cudaStreamWaitEvent(comm->streams_[idx], comm->events_[idx], 0));

      PADDLE_ENFORCE(cudaEventDestroy(comm->events_[idx]));

      PADDLE_ENFORCE(ncclCommDestroy(comm->comms_[idx]));
    }
    comm.reset(nullptr);
  }
}

Communicator* NCCLManager::GetCommunicator(const std::vector<int>& gpus) {
  std::string key;
  for (auto& id : gpus) {
    key += std::to_string(id);
  }
  std::sort(key.begin(), key.end());

  std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);

  auto it = comm_table.find(key);

  if (it->second == nullptr) {
    auto* comm = new Communicator(gpus);
    PADDLE_ENFORCE(
        ncclCommInitAll(comm->comms_.data(), gpus.size(), gpus.data()));

    for (size_t i = 0; i < gpus.size(); ++i) {
      platform::SetDeviceId(gpus[i]);

      // block wait
      PADDLE_ENFORCE(cudaEventCreateWithFlags(
          &comm->events_[i], cudaEventBlockingSync | cudaEventDisableTiming));
    }
    comm_table[key].reset(comm);
  }
  return comm_table[key].get();
}

}  // namespace operators
}  // namespace paddle
