#ifndef PADDLE_OPTIMIZER_TENSOR_H_
#define PADDLE_OPTIMIZER_TENSOR_H_
/**
 * @brief tensor used by optimizer
 */

#include <string.h>
#include "paddle/math/BaseMatrix.h"

namespace paddle {
namespace optimizer {

template <class T>
using TensorBase = BaseMatrixT<T>;

template <class T>
class TensorT : public TensorBase<T> {
public:
  TensorT(T* data, int size) : TensorBase<T>(1, size, 0, data, false, false) {}
  TensorT(const TensorT& t)
      : TensorBase<T>(1, t.size(), 0, t.get_buffer(), false, false) {}
  TensorT& operator=(const TensorT& t) {
    this->size_ = t.size();
    this->data_ = t.get_buffer();
  }
  T* get_buffer() { return this->data_; }
  T& operator[](const int idx) {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return this->data_[idx];
  }
  // TODO: replace with tensorshape
  size_t size() const { return this->width_; }
};

// TODO(zhihong): design problem of dynamic datatype, need to fix
typedef TensorT<real> Tensor;

}  // namespace optimizer
}  // namespace paddle

#endif
