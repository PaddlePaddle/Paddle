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

#include <sstream>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace platform {

template <>
std::string json_vector<std::string>(
    const std::vector<std::string> type_vector) {
  std::ostringstream res_stream;
  auto count = type_vector.size();
  res_stream << "[";
  for (auto it = type_vector.begin(); it != type_vector.end(); it++) {
    if (count > 1) {
      res_stream << "\"" << (*it) << "\""
                 << ",";
    } else {
      res_stream << "\"" << (*it) << "\"";
    }
    count--;
  }
  res_stream << "]";
  return res_stream.str();
}

#ifdef PADDLE_WITH_CUPTI

#ifdef PADDLE_WITH_HIP

#include "hip/hip_runtime.h"
float CalculateEstOccupancy(uint32_t DeviceId,
                            int32_t DynamicSharedMemory,
                            int32_t BlockX,
                            int32_t BlockY,
                            int32_t BlockZ,
                            void* kernelFunc,
                            uint8_t launchType) {
  float occupancy = 0.0;
  std::vector<int> device_ids = GetSelectedDevices();
  if (DeviceId < device_ids.size()) {
    const gpuDeviceProp& device_property = GetDeviceProperties(DeviceId);
    int blockSize = BlockX * BlockY * BlockZ;
    int numBlock = 0;
    hipError_t status;
    if (launchType == 0) {
      status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlock, kernelFunc, blockSize, DynamicSharedMemory);
      if (status == hipSuccess) {
        occupancy = static_cast<double>(numBlock) * blockSize /
                    device_property.maxThreadsPerMultiProcessor;
      } else {
        LOG(WARNING) << "Failed to calculate estimated occupancy, status = "
                     << status << std::endl;
      }
    } else if (launchType == 100) {
      status = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlock,
          reinterpret_cast<hipFunction_t>(kernelFunc),
          blockSize,
          DynamicSharedMemory);
      if (status == hipSuccess) {
        occupancy = static_cast<double>(numBlock) * blockSize /
                    device_property.maxThreadsPerMultiProcessor;
      } else {
        LOG(WARNING) << "Failed to calculate estimated occupancy, status = "
                     << status << std::endl;
      }
    } else {
      LOG(WARNING) << "Failed to calculate estimated occupancy, can not "
                      "recognize launchType : "
                   << launchType << std::endl;
    }
  }
  return occupancy;
}

#else

float CalculateEstOccupancy(uint32_t DeviceId,
                            uint16_t RegistersPerThread,
                            int32_t StaticSharedMemory,
                            int32_t DynamicSharedMemory,
                            int32_t BlockX,
                            int32_t BlockY,
                            int32_t BlockZ,
                            float BlocksPerSm) {
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
    cudaOccError status =
        cudaOccMaxActiveBlocksPerMultiprocessor(&occ_result,
                                                &prop,
                                                &occFuncAttr,
                                                &occDeviceState,
                                                blockSize,
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
#endif  // PADDLE_WITH_HIP

#endif  // PADDLE_WITH_CUPTI

const char* StringTracerMemEventType(TracerMemEventType type) {
  static const char* categary_name_[] = {
      "Allocate", "Free", "ReservedAllocate", "ReservedFree"};
  return categary_name_[static_cast<int>(type)];
}

const char* StringTracerEventType(TracerEventType type) {
  static const char* categary_name_[] = {"Operator",
                                         "Dataloader",
                                         "ProfileStep",
                                         "CudaRuntime",
                                         "Kernel",
                                         "Memcpy",
                                         "Memset",
                                         "UserDefined",
                                         "OperatorInner",
                                         "Forward",
                                         "Backward",
                                         "Optimization",
                                         "Communication",
                                         "PythonOp",
                                         "PythonUserDefined",
                                         "MluRuntime"};
  return categary_name_[static_cast<int>(type)];
}

}  // namespace platform
}  // namespace paddle
