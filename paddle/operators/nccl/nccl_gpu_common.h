#pragma once
#include <nccl.h>

#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

class NCCLManager {
 public:
  static NCCLManager* Get() {
    static NCCLManager m;
    return &m;
  }

  NCCLManager() { _comms.resize(_gpu_worlds.size()); }
  ~NCCLManager() {}

 private:
  // clang-format off
  std::vector<ncclComm_t> _comms;
  std::vector<int> _gpu_worlds;
  // clang-format on
};

class NCCLContext : public DeviceContext {
 public:
  explicit NCCLContext(GPUPlace place);
  virtual ~NCCLContext();

 private:
  // clang-format off
  std::vector<int> _gpu_ids;
  std::vector<cudaStream_t> _streams;
  int root_gpu;
  // clang-format on
};
}
}
