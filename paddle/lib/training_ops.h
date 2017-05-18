#ifndef __PADDLE_TRAINING_OPS_H__
#define __PADDLE_TRAINING_OPS_H__
/*! \brief this file contains the optimizer algorithnm as ops
  name convention. applyXXX  e.g applyGradientDescent
 */
// #include "math/Tensor.h"
#include "Tensor.h"

template<typename T>
void applyGradientDescent(Tensor<T> &parameter,
                          const Tensor<T> &gradient,
                          double learning_rate);

template<typename T>
void applyMomentum(Tensor<T> &parameter,
                          Tensor<T> &momentum,
                          const Tensor<T> &gradient,
                          double learning_rate,
                          double mu,
                          double weight_decay);

template<typename T>
void applyAdam(Tensor<T> &parameter,
               const Tensor<T> &gradient,
               double learning_rate);


#endif
