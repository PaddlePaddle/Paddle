#pragma once
/**
 * @brief tensor used by optimizer
 */

#include <string.h>
#include <memory>
#include "paddle/math/MemoryHandle.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Logging.h"

namespace paddle {
namespace optimizer {

template <class T>
class TensorT {
public:
  TensorT(size_t size)
      : TensorT(std::make_shared<CpuMemoryHandle>(size * sizeof(float)), size) {
  }
  TensorT(CpuMemHandlePtr handle, size_t size)
      : height_(1),
        width_(size),
        data_(reinterpret_cast<T*>(handle->getBuf())) {}

  TensorT(T* data, size_t size) : height_(1), width_(size), data_(data) {}

  TensorT(T* data, size_t h, size_t w) : height_(h), width_(w), data_(data) {}

  virtual ~TensorT() {}

  T* get_buffer() { return this->data_; }

  T& operator[](const size_t idx) {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  T& operator[](const size_t idx) const {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  // TODO: replace with tensorshape
  size_t size() const { return this->width_ * this->height_; }

protected:
  size_t height_;
  size_t width_;
  T* data_;
};

// TODO(zhihong): design problem of dynamic datatype, need to fix it
typedef TensorT<float> Tensor;

}  // namespace optimizer
}  // namespace paddle
