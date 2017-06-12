#include "paddle/majel/malloc.h"
#include <glog/logging.h>
#include <memory>

#ifndef PADDLE_ONLY_CPU
#include "cuda_runtime.h"
#endif

namespace majel {
namespace malloc {
namespace detail {
#ifndef PADDLE_ONLY_CPU
const char* get_cuda_error_string() {
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}

const char* get_cuda_error_string(size_t err) {
  return cudaGetErrorString((cudaError_t)err);
}

void* malloc_cuda(size_t size) {
  void* dest_d;
  cudaError_t result = cudaMalloc((void**)&dest_d, size);
  if (result == cudaSuccess) {
    return dest_d;
  }

  cudaGetLastError();
  return nullptr;
}

void free_cuda(void* dest_d) {
  CHECK_NOTNULL(dest_d);

  cudaError_t err = cudaFree(dest_d);
  CHECK(cudaSuccess == err || cudaErrorCudartUnloading == err)
      << get_cuda_error_string();
}

void set_cuda_device(int device) { cudaSetDevice(device); }
#endif

class DefaultAllocator {
public:
  static void* malloc(majel::Place place, size_t size);

  static void free(majel::Place, void* ptr);
};

class DefaultAllocatorMallocVisitor : public boost::static_visitor<void*> {
public:
  DefaultAllocatorMallocVisitor(size_t size) : size_(size) {}

  void* operator()(majel::CpuPlace p) {
    void* address;
    CHECK_EQ(posix_memalign(&address, 32ul, size_), 0);
    return address;
  }

#ifndef PADDLE_ONLY_CPU
  void* operator()(majel::GpuPlace p) {
    set_cuda_device(p.device);
    void* address = malloc_cuda(size_);
    return address;
  }
#endif

private:
  size_t size_;
};

class DefaultAllocatorFreeVisitor : public boost::static_visitor<void> {
public:
  DefaultAllocatorFreeVisitor(void* ptr) : ptr_(ptr) {}
  void operator()(majel::CpuPlace p) {
    if (ptr_) {
      ::free(ptr_);
    }
  }

#ifndef PADDLE_ONLY_CPU
  void operator()(majel::GpuPlace p) {
    set_cuda_device(p.device);
    if (ptr_) {
      free_cuda(ptr_);
    }
  }
#endif

private:
  void* ptr_;
};

void* DefaultAllocator::malloc(majel::Place place, size_t size) {
  DefaultAllocatorMallocVisitor visitor(size);
  return boost::apply_visitor(visitor, place);
}

void DefaultAllocator::free(majel::Place place, void* ptr) {
  DefaultAllocatorFreeVisitor visitor(ptr);
  boost::apply_visitor(visitor, place);
}

}  // namespace detail
}  // namespace malloc
}  // namespace majel
namespace majel {
namespace malloc {

void* malloc(majel::Place place, size_t size) {
  return detail::DefaultAllocator::malloc(place, size);
}

void free(majel::Place place, void* ptr) {
  detail::DefaultAllocator::free(place, ptr);
}
}  // namespace malloc
}  // namespace majel
