#pragma once
#include <nccl.h>

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

#define NCCL_CHECK(condition)                                             \
  do {                                                                    \
    ncclResult_t ret = (condition);                                       \
    PADDLE_ENFORCE(ret == ncclSuccess, "Error invoking NCCL: ", __FILE__, \
                   __LINE__, ncclGetErrorString(ret));                    \
  } while (0)

class WaitGroup {
 public:
  inline void Add(int n) {
    std::unique_lock<std::mutex> lk(mu_);
    PADDLE_ENFORCE(n >= 0, "add wait must >=0.");
    counter_ += n;
  }

  inline void Done(int n) {
    std::unique_lock<std::mutex> lk(mu_);
    PADDLE_ENFORCE(n <= counter_, " wait group done unmatch to add.");
    counter_ -= n;
    if (counter_ == 0) {
      cv_.notify_all();
    }
  }

  inline void Add() { Add(1); }

  inline void Done() { Done(1); }

  inline void Wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return counter_ == 0; });
  }

  inline int GetCount() {
    std::unique_lock<std::mutex> lk(mu_);
    return counter_;
  }

 private:
  int counter_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
};

// class NCCLContext : public DeviceContext {
// public:
//   explicit NCCLContext(GPUPlace place);
//   virtual ~NCCLContext();

// private:
//   std::vector<int> gpu_ids_;
//   std::vector<cudaStream_t> streams_;
// };

// TODO(dzh) : make resources managed unified with framework
struct Communicator {
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t*> streams_;
  std::vector<cudaEvent_t> events_;
  std::vector<int> gpus_;
  WaitGroup wg_;
  int root_gpu = -1;
  // cudaEvent_t root_monitor;
  explicit Communicator(const std::vector<int>& gpus) : gpus_(gpus) {
    comms_.resize(gpus.size());
    streams_.resize(gpus.size());
    events_.resize(gpus.size());
  }
  // Communicator(int num_device): comms_.resize(num_device) {}

  inline int get_root_gpu() const { return root_gpu; }

  inline void set_root_gpu(int id) { root_gpu = id; }
};

class NCCLManager {
 public:
  static NCCLManager* Get() {
    static NCCLManager m;
    return &m;
  }

  NCCLManager();

  ~NCCLManager();

  // for each card only have one communicator
  Communicator* GetCommunicator(const std::vector<int>& gpus) const;

 private:
  // // the gpu id list available. Note that only support
  // // whole world communication.
  // std::vector<int> _gpu_worlds;

  // communicator list
  std::unordered_map<std::string /* key*/, Communicator*> comm_table;
};

}  // namespace operators
}  // namespace paddle
