#ifndef PADDLE_OPTIMIZER_FACTORY_H_
#define PADDLE_OPTIMIZER_FACTORY_H_

#include "parameter_optimizer.h"
#include "training_ops.h"

namespace paddle {
namespace optimizer {

template <class T>
class SGDOptimizer : public ParameterOptimizer {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
    auto parameter = &(*parameter_.get());
    learning_rate = applyLinearLearningRate(config_);
    applyGradientDescent(parameter, gradient, learning_rate);
  }
};

template <class T>
class MomentumOptimizer : public ParameterOptimizer {
public:
  /*! \brief call the applyXX for example  */
  MomentumOptimizer(const paddle::optimizer_config &config);
  void update(const Tensor<T> &gradient) {
    learning_rate = applyExpLearningRate(config_);
    applyMomentum(
        parameter, gradient, momentum, learning_rate, mu, weight_decay);
  }

private:
  double momentum;
};

}  // namespace optimizer
}  // namespace paddle
#endif
