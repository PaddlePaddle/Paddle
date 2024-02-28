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

std::variant<int, std::array<int, 3>> CUDABackendAPI::get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id) {
  int dev_index = device_id ? device_id.value() : this->now_device_id;
  std::variant<int, std::array<int, 3>> rv_variant;
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDims: {
      cudaDeviceProp prop_;
      CUDA_CALL(cudaGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{prop_.maxThreadsDim[0], prop_.maxThreadsDim[1], prop_.maxThreadsDim[2]};
      break;
    }
    case DeviceProperty::MaxGridDims: {
      cudaDeviceProp prop_;
      CUDA_CALL(cudaGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{prop_.maxGridSize[0], prop_.maxGridSize[1], prop_.maxGridSize[2]};
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, dev_index));
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty:: MaxBlocksPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::WarpSize: {
      CUDA_CALL(cudaDeviceGetAttribute(&rv, cudaDeviceAttr::cudaDevAttrWarpSize, dev_index));
      rv_variant = rv;
      break;
    }
    default:
      LOG(FATAL) << "Not supported device property!";
  }
  return rv_variant;
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

void CUDABackendAPI::stream_sync(void* stream){
  CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn