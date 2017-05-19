#ifndef PADDLE_FAKE_TENSOR_H_
#define PADDLE_FAKE_TENSOR_H_
/*! \brief fake tensor for testing */

#include "math/BaseMatrix.h"
using TensorBase = BaseMatrix<T>;

class Tensor: public TensorBase {
  
}


#endif
