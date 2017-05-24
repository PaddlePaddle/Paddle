#include "paddle/majel/allocation.h"
#include <boost/variant.hpp>
#include "paddle/majel/malloc.h"

namespace majel {
namespace detail {

class Allocator : public boost::static_visitor<void*> {
public:
  Allocator(size_t size) : size_(size) {}

  void* operator()(const CpuPlace& p) const {
    void* address = majel::malloc::malloc(p, size_);
    return address;
  }

  void* operator()(const GpuPlace& p) const {
    void* address = majel::malloc::malloc(p, size_);
    return address;
  }

private:
  size_t size_;
};

class Deallocator : public boost::static_visitor<> {
public:
  Deallocator(void* ptr) : ptr_(ptr) {}

  void operator()(CpuPlace p) const {
    if (ptr_) {
      majel::malloc::free(p, ptr_);
    }
  }

  void operator()(GpuPlace p) const {
    if (ptr_) {
      majel::malloc::free(p, ptr_);
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
