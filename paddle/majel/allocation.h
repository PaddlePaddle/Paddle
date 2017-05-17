#pragma once
#include <memory>

#include <majel/place.h>

namespace majel {

class Allocation {
public:
  Allocation();
  Allocation(size_t size);
  Allocation(size_t size, Place place);

  // Creates a non-owned allocation (an allocation not owned by the Majel
  // memory allocator); non-owned allocations are not cleaned up in the
  // destructor.
  Allocation(void* ptr, size_t size, Place place);

  ~Allocation();
  // No copying!
  Allocation(const Allocation&) = delete;
  // No assigning!
  Allocation& operator=(const Allocation&) = delete;

  void* ptr() const;
  void* end() const;
  Place place() const;
  size_t size() const;

private:
  bool owned_;
  void* ptr_;
  size_t size_;
  Place place_;
};

}  // namespace majel
