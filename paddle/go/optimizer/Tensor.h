#ifndef PADDLE_FAKE_TENSOR_H_
#define PADDLE_FAKE_TENSOR_H_
/*! \brief fake tensor for testing */

#include "paddle/math/BaseMatrix.h"

namespace paddle {
template <class T>
using TensorBase = BaseMatrixT<T>;

template <class T>
class Tensor : public TensorBase<T> {
public:
  // Tensor(T* data, int size) :
  //   height_(size), width_(1), stride_(0), data_(data), trans_(false),
  //   useGpu(false) {};
  Tensor(T* data, int size) : TensorBase<T>(size, 1, 0, data, false, false) {}
  T* get_buffer() { return this->data_; }
};

#endif
