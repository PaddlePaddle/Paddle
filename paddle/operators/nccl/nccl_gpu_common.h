#pragma once
#include <nccl.h>

#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>

#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {


// class NCCLContext : public DeviceContext {
// public:
//   explicit NCCLContext(GPUPlace place);
//   virtual ~NCCLContext();

// private:
//   std::vector<int> gpu_ids_;
//   std::vector<cudaStream_t> streams_;
// };


class Communicator;

class NCCLManager {
 public:
  static NCCLManager* Get() {
    static NCCLManager m;
    return &m;
  }

  NCCLManager() {
  }
  ~NCCLManager() {}

  // for each card only have one communicator
  Communicator* GetCommunicator() const;

 private:
  struct Communicator {
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t*> streams_; // do not own
    std::vector<cudaEvent_t> events_;
    int root_gpu;
  };

  // the gpu id list available. Note that only support
  // whole world communication.
  std::vector<int> _gpu_worlds;

  // communicator list
  std::unordered_map<std::string /* key*/, Communicator*> comms_;
};

}  // namespace operators
}  // namespace paddle
