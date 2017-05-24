#pragma once

#include "paddle/majel/allocation.h"
#include "paddle/majel/place.h"

namespace majel {

class Buffer {
public:
  Buffer()
      : external_address_(nullptr),
        allocation_(std::make_shared<Allocation>(0)) {}
  Buffer(void* address)
      : external_address_(address),
        allocation_(std::make_shared<Allocation>(0)) {}
  Buffer(void* address, Place p)
      : external_address_(address),
        allocation_(std::make_shared<Allocation>(0, p)) {}
  Buffer(std::shared_ptr<Allocation> allocation)
      : external_address_(nullptr), allocation_(allocation) {}

public:
  void* get_address() const {
    if (allocation_->ptr() == nullptr) {
      return external_address_;
    }

    return allocation_->ptr();
  }

  Place get_place() const { return allocation_->place(); }

  std::shared_ptr<Allocation> data() const { return allocation_; }

private:
  void* external_address_;
  std::shared_ptr<Allocation> allocation_;
};

}  // namespace majel
