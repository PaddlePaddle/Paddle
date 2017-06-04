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
class Tensor : public TensorBase<T> {
public:
  Tensor(T* data, int size) : TensorBase<T>(size, 1, 0, data, false, false) {}
  T* get_buffer() { return this->data_; }
  // TODO: replace with tensorshape
  size_t width() { return this->width_; }
};

}  // namespace optimizer
}  // namespace paddle

#endif
