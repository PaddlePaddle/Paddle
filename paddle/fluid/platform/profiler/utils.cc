/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/utils.h"

#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace platform {
#ifdef PADDLE_WITH_CUPTI
float CalculateEstOccupancy(uint32_t DeviceId, uint16_t RegistersPerThread,
                            int32_t StaticSharedMemory,
                            int32_t DynamicSharedMemory, int32_t BlockX,
                            int32_t BlockY, int32_t BlockZ, float BlocksPerSm) {
  float occupancy = 0.0;
  std::vector<int> device_ids = GetSelectedDevices();
  if (DeviceId < device_ids.size()) {
    const gpuDeviceProp& device_property = GetDeviceProperties(DeviceId);
    cudaOccFuncAttributes occFuncAttr;
    occFuncAttr.maxThreadsPerBlock = INT_MAX;
    occFuncAttr.numRegs = RegistersPerThread;
    occFuncAttr.sharedSizeBytes = StaticSharedMemory;
    occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
    occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
    occFuncAttr.maxDynamicSharedSizeBytes = 0;
    const cudaOccDeviceState occDeviceState = {};
    int blockSize = BlockX * BlockY * BlockZ;
    size_t dynamicSmemSize = DynamicSharedMemory;
    cudaOccResult occ_result;
    cudaOccDeviceProp prop(device_property);
    cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
        &occ_result, &prop, &occFuncAttr, &occDeviceState, blockSize,
        dynamicSmemSize);
    if (status == CUDA_OCC_SUCCESS) {
      if (occ_result.activeBlocksPerMultiprocessor < BlocksPerSm) {
        BlocksPerSm = occ_result.activeBlocksPerMultiprocessor;
      }
      occupancy =
          BlocksPerSm * blockSize /
          static_cast<float>(device_property.maxThreadsPerMultiProcessor);
    } else {
      LOG(WARNING) << "Failed to calculate estimated occupancy, status = "
                   << status << std::endl;
    }
  }
  return occupancy;
}
#endif

}  // namespace platform
}  // namespace paddle
