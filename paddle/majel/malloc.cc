#include "malloc.h"
#include <glog/logging.h>

#ifndef PADDLE_ONLY_CPU
#include <cuda_runtime.h>
#endif

#define CHECK_CUDA(cudaFunc)                                         \
  do {                                                               \
    cudaError_t cudaStat = cudaFunc;                                 \
    CHECK_EQ(cudaSuccess, cudaStat) << "Cuda Error: "                \
                                    << cudaGetErrorString(cudaStat); \
  } while (0)

namespace majel {
namespace malloc {
namespace detail {
#ifndef PADDLE_ONLY_CPU
const char* hl_get_device_error_string() {
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}

const char* hl_get_device_error_string(size_t err) {
  return cudaGetErrorString((cudaError_t)err);
}

void* hl_malloc_device(size_t size) {
  void* dest_d;

  CHECK(size) << __func__ << ": the size for device memory is 0, please check.";
  CHECK_CUDA(cudaMalloc((void**)&dest_d, size));

  return dest_d;
}

void hl_free_mem_device(void* dest_d) {
  CHECK_NOTNULL(dest_d);

  cudaError_t err = cudaFree(dest_d);
  CHECK(cudaSuccess == err || cudaErrorCudartUnloading == err)
      << hl_get_device_error_string();
}
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
    CHECK(address) << "Fail to allocate CPU memory: size=" << size_;
    return address;
  }

#ifndef PADDLE_ONLY_CPU
  void* operator()(majel::GpuPlace p) {
    void* address = hl_malloc_device(size_);
    CHECK(address) << "Fail to allocate GPU memory " << size_ << " bytes";
    return address;
  }
#else
  void* operator()(majel::GpuPlace p) {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
    return nullptr;
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
    if (ptr_) {
      hl_free_mem_device(ptr_);
    }
  }

#else
  void operator()(majel::GpuPlace p) {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
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
