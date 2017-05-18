#include "training_ops.h"

/*! \brief implement different update method
e.g. applyGradientDescentAvx
 */

template<typename T>
void applyGradientDescent(Tensor<T> &parameter,
                          Tensor<T> &momentum,
                          const Tensor<T> &gradient,
                          double learning_rate,
                          double mu,
                          double weight_decay) {
  weight_decay *= learning_rate;
  /*! \brief TODO(will replace with matrix dot) */
  for(size_t i=0; i < parameter.size(); ++i) {
    momentum[i] = mu * momentum[i] - learning_rate * gradient[i] - weight_decay * parameter[i];
    parameter[i] += momentum[i];
  }
}

template<typename T>
void applyGradientDescent(Tensor<T> &parameter,
                          const Tensor<T> &gradient,
                          double learning_rate) {
  /*! \brief TODO(will replace with matrix dot) */
  for(size_t i=0; i < parameter.size(); ++i) {
    parameter[i] -= gradient[i] * learning
    }
}
