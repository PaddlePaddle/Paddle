#include <boost/variant.hpp>

#include <majel/allocation.h>
#include "hl_gpu.h"
#include "paddle/utils/Logging.h"
namespace majel {
namespace detail {

class Allocator : public boost::static_visitor<void*> {
public:
  Allocator(size_t size) : size_(size) {}

  void* operator()(const CpuPlace& p) const {
    void* address;
    CHECK_EQ(posix_memalign(&address, 32ul, size_), 0);
    CHECK(address) << "Fail to allocate CPU memory: size=" << size_;
    return address;
  }

  void* operator()(const GpuPlace& p) const {
    void* address = hl_malloc_device(size_);
    CHECK(address) << "Fail to allocate GPU memory " << size_ << " bytes";
    return address;
  }

private:
  size_t size_;
};

class Deallocator : public boost::static_visitor<> {
public:
  Deallocator(void* ptr) : ptr_(ptr) {}

  void operator()(CpuPlace p) const {
    void* ptr = ptr_;
    if (ptr) {
      ::free(ptr);
    }
  }

  void operator()(GpuPlace p) const {
    void* ptr = ptr_;
    if (ptr) {
      hl_free_mem_device(ptr);
    }
  }

private:
  void* ptr_;
};

}  // namespace detail
}  // namespace majel

namespace majel {

Allocation::Allocation() : Allocation(0, get_place()) {}

Allocation::Allocation(size_t size) : Allocation(size, get_place()) {}

Allocation::Allocation(size_t size, Place place)
    : owned_(true), size_(size), place_(place) {
  if (size > 0) {
    majel::detail::Allocator allocator(size_);
    ptr_ = boost::apply_visitor(allocator, place_);
    if (ptr_ == nullptr) {
      throw std::bad_alloc();
    }
  } else {
    ptr_ = nullptr;
  }
}

Allocation::Allocation(void* ptr, size_t size, Place place)
    : owned_(false), ptr_(ptr), size_(size), place_(place) {}

Allocation::~Allocation() {
  // If we don't own this allocation don't try to deallocate it
  if (!owned_) {
    return;
  }

  if (ptr_ != nullptr) {
    majel::detail::Deallocator deallocator(ptr_);

    boost::apply_visitor(deallocator, place_);
  }
}

void* Allocation::ptr() const { return ptr_; }

void* Allocation::end() const { return (uint8_t*)ptr_ + size_; }

size_t Allocation::size() const { return size_; }

Place Allocation::place() const { return place_; }

}  // namespace majel
