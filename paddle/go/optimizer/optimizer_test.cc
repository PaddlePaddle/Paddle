#include "training_ops.h"

namespace paddle {

template <class T>
class OpitmizerTests {
private:
  Tensor<T> parameter;
  Tensor<T> gradient;
};

void applyGradientDescent_TEST() {}
}  // namespace paddle
