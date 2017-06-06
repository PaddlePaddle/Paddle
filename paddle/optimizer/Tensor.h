#ifndef PADDLE_OPTIMIZER_TENSOR_H_
#define PADDLE_OPTIMIZER_TENSOR_H_
/**
 * @brief tensor used by optimizer
 */

#include <string.h>
#include "paddle/utils/Common.h"
#include "paddle/utils/Logging.h"

namespace paddle {
namespace optimizer {

template <class T>
class TensorT {
public:
  TensorT(size_t h, size_t w, T* data) : height_(h), width_(w), data_(data_) {}
  TensorT(T* data, int size) : height_(1), width_(size), data_(data) {}
  TensorT(const TensorT& t)
      : TensorT(1, t.size(), 0, t.get_buffer(), false, false) {}
  TensorT& operator=(const TensorT& t) {
    this->width_ = t.size();
    this->data_ = t.get_buffer();
  }
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
typedef TensorT<real> Tensor;

}  // namespace optimizer
}  // namespace paddle

#endif
