#include "paddle/cinn/runtime/hip/hip_backend_api.h"
#include <hip/hip_runtime.h>
#include "paddle/cinn/runtime/hip/hip_util.h"
#include <glog/logging.h>

namespace cinn {
namespace runtime {
namespace hip {
HIPBackendAPI* HIPBackendAPI::Global() {
  static auto* inst = new HIPBackendAPI();
  return inst;
}

void HIPBackendAPI::set_device(int device_id) {
  HIP_CALL(hipSetDevice(device_id));
  this->now_device_id = device_id;
}

int HIPBackendAPI::get_device() {
  int device_id = 0;
  HIP_CALL(hipGetDevice(&device_id));
  return device_id;
}

std::variant<int, std::array<int, 3>> HIPBackendAPI::get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id) {
  int dev_index = device_id.value_or(this->now_device_id);
  std::variant<int, std::array<int, 3>> rv_variant;
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDims: {
      hipDeviceProp_t prop_;
      HIP_CALL(hipGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{prop_.maxThreadsDim[0], prop_.maxThreadsDim[1], prop_.maxThreadsDim[2]};
      break;
    }
    case DeviceProperty::MaxGridDims: {
      hipDeviceProp_t prop_;
      HIP_CALL(hipGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{prop_.maxGridSize[0], prop_.maxGridSize[1], prop_.maxGridSize[2]};
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerMultiProcessor, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeMultiprocessorCount, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty:: MaxBlocksPerSM: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerMultiProcessor, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::WarpSize: {
      HIP_CALL(hipDeviceGetAttribute(&rv, hipDeviceAttribute_t::hipDeviceAttributeWarpSize, dev_index));
      rv_variant = rv;
      break;
    }
    default:
      LOG(FATAL) << "Not supported device property!";
  }
  return rv_variant;
}


void* HIPBackendAPI::malloc(size_t numBytes){
  void* dev_mem = nullptr;
  HIP_CALL(hipMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void HIPBackendAPI::free(void* data) {
  HIP_CALL(hipFree(data));
}

void HIPBackendAPI::memset(void* data, int value, size_t numBytes){
  HIP_CALL(hipMemset(data, value, numBytes));
}

void HIPBackendAPI::memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type){
  hipMemcpyKind copy_kind;
  switch(type){
    case MemcpyType::HostToHost:
      copy_kind = hipMemcpyHostToHost;
      break;
    case MemcpyType::HostToDevice:
      copy_kind = hipMemcpyHostToDevice;
      break;
    case MemcpyType::DeviceToHost:
      copy_kind = hipMemcpyDeviceToHost;
      break;
    case MemcpyType::DeviceToDevice:
      copy_kind = hipMemcpyDeviceToDevice;
      break;
  }
  HIP_CALL(hipMemcpy(dest, src, numBytes, copy_kind));
}

void HIPBackendAPI::device_sync(){
  HIP_CALL(hipDeviceSynchronize());
}

void HIPBackendAPI::stream_sync(void* stream){
  HIP_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
}

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
