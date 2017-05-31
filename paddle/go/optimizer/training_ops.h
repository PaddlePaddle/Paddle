#ifndef PADDLE_LIB_TRAINING_OPS_H_
#define PADDLE_LIB_TRAINING_OPS_H_
/*! \brief this file contains the optimizer algorithnm as ops
  name convention. applyXXX  e.g applyGradientDescent
 */

#include <algorithm>
#include "Tensor.h"

namespace paddle {
namespace optimizer {


template <typename T>
void applyGradientDescent(const Tensor<T> &gradient);

template <typename T>
void applyMomentum(const Tensor<T> &gradient, double mu, double weight_decay);

template <typename T>
void applyAdam(const Tensor<T> &gradient, double mu, double learning_rate);
}  // namespace optimizer
}  // namespace paddle

#endif
