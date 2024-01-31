#include "paddle/cinn/runtime/cuda/cuda_backend_api.h"
#include <glog/logging.h>
namespace cinn {
namespace runtime {
namespace cuda {
CUDABackendAPI* CUDABackendAPI::Global() {
  static auto* inst = new CUDABackendAPI();
  return inst;
}

void CUDABackendAPI::set_device(int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
  this->now_device_id = device_id;
}

int CUDABackendAPI::get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id) {
  int index = device_id ? device_id.value() : this->now_device_id;
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDims: {
      LOG(FATAL) << "Not supported device property!";
      break;
    }
    case DeviceProperty::MaxGridDims: {
      LOG(FATAL) << "Not supported device property!";
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock, index));
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, index));
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, index));
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, index));
      break;
    }
    case DeviceProperty:: MaxBlocksPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, index));
      break;
    }
    case DeviceProperty::WarpSize: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrWarpSize, index));
      break;
    }
    default:
      LOG(FATAL) << "Not supported device property!";
  }
  return rv;
}

void* CUDABackendAPI::malloc(size_t numBytes){
  void* dev_mem = nullptr;
  CUDA_CALL(cudaMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void CUDABackendAPI::free(void* data) {
  CUDA_CALL(cudaFree(data));
}

void CUDABackendAPI::memset(void* data, int value, size_t numBytes){
  CUDA_CALL(cudaMemset(data, value, numBytes));
}

void CUDABackendAPI::memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type){
  cudaMemcpyKind copy_kind;
  switch(type){
    case MemcpyType::HostToHost:
      copy_kind = cudaMemcpyHostToHost;
      break;
    case MemcpyType::HostToDevice:
      copy_kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyType::DeviceToHost:
      copy_kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyType::DeviceToDevice:
      copy_kind = cudaMemcpyDeviceToDevice;
      break;
  }
  CUDA_CALL(cudaMemcpy(dest, src, numBytes, copy_kind));
}

void CUDABackendAPI::device_sync(){
  CUDA_CALL(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn