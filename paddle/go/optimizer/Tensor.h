#ifndef PADDLE_FAKE_TENSOR_H_
#define PADDLE_FAKE_TENSOR_H_
/*! \brief fake tensor for testing */

#include "math/BaseMatrix.h"
#include "string.h"
namespace paddle {
template <class T>
using TensorBase = BaseMatrixT<T>;

template <class T>
class Tensor : public TensorBase {
public:
  Tensor(T* data, int size) : TensorBase(size, 1, data, true, false);
  T* get_buffer() { return this->data_; }
}

#endif
