#ifndef PADDLE_FAKE_TENSOR_H_
#define PADDLE_FAKE_TENSOR_H_
/**
 * @brief fake tensor for testing 
 */

#include "paddle/math/BaseMatrix.h"
#include <string.h>

namespace paddle {
template <class T>
using TensorBase = BaseMatrixT<T>;

template <class T>
class Tensor : public TensorBase<T> {
public:
  Tensor(T* data, int size) : TensorBase<T>(size, 1, 0, data, false, false) {}
  T* get_buffer() { return this->data_; }
  // TODO: replace with tensorshape
  size_t height() {
    return this->height_;
  }
};

#endif
