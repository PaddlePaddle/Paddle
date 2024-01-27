#include "paddle/cinn/runtime/cuda/cuda_backend_api.h"
namespace cinn {
namespace runtime {
namespace cuda {
CUDABackendAPI* CUDABackendAPI::Global() {
  static auto* inst = new CUDABackendAPI();
  return inst;
}

void CUDABackendAPI::set_device(int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
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