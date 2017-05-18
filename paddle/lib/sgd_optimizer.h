#ifndef __PADDLE_SGD_OPTIMIZER_H__
#define __PADDLE_SGD_OPTIMIZER_H__

#include "parameter_optimizer.h"
#include "training_ops.h"

class SGDOptimizer : public ParameterOptimizer {
  public:
  /*! \brief call the applySGD for example  */
  void update(Tensor<T> &parameter,
              const Tensor<T> &gradient,
              double learning_rate) {
    applyGradientDescent(parameter, gradient, learning_rate);
  }
};

class MomentumOptimizer : public ParameterOptimizer {

public:
  /*! \brief call the applyXX for example  */
  void update(Tensor<T> &parameter,
              Tensor<T> &momentum,
              const Tensor<T> &gradient,
              double learning_rate,
              double mu,
              double weight_decay) {
    applyMomentum(parameter, gradient, momentum, learning_rate, mu, weight_decay);
  }
  
}


#endif
