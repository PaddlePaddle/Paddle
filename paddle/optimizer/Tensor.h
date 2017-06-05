#ifndef PADDLE_OPTIMIZER_TENSOR_H_
#define PADDLE_OPTIMIZER_TENSOR_H_
/**
 * @brief tensor used by optimizer
 */

#include <string.h>
#include "optimizer.h"
#include "paddle/math/BaseMatrix.h"

namespace paddle {
namespace optimizer {

template <class T>
using TensorBase = BaseMatrixT<T>;

template <class T>
class Tensor : public TensorBase<T> {
public:
  Tensor(T* data, int size) : TensorBase<T>(1, size, 0, data, false, false) {}
  T* get_buffer() { return this->data_; }
  T& operator[](const int idx) {
    CHECK(idx >= 0 && idx < this->width_) << " out of index range";
    return this->data_[idx];
  }
  // TODO: replace with tensorshape
  size_t size() const { return this->width_; }
};

}  // namespace optimizer
}  // namespace paddle

#endif
