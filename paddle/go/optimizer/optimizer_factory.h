#ifndef PADDLE_OPTIMIZER_FACTORY_H_
#define PADDLE_OPTIMIZER_FACTORY_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {

static double applyLinearLearningRate(paddle::optimizer_config &config) {
  config.learning_rate =
      std::max(config.learning_rate - config.decay_a * config.samples_processed,
               config.decay_b);
  return config.learning_rate;
}

static double applyExpLearningRate(paddle::optimizer_config &config) {
  double decayRatio = (double)config.samples_processed / config.decay_b;
  config.learning_rate = config.learning_rate * std::pow(decay_a, decayRatio);
}

// double applyLearningRate(double learning_rate, uint32_t epoch);


// double applyLinearLearningRate(double learning_rate, double decay_a, double
// decay_b, uint64_t samples_processed);

// double applyExpLearningRate(double learning_rate, double decay_a, double
// decay_b, uint64_t samples_processed);

// double applyPolyLearningRate(double learning_rate, double decay_a, double
// decay_b, uint64_t samples_processed, uint32_t epoch);

// double applyCaffePolyLearningRate(double learning_rate, double decay_a,
// double decay_b, uint64_t samples_processed, uint32_t epoch);

template <class T>
class SGDOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  SGDOptimizer(const OptimizerConfig &config);
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  void update(const Tensor<T> &gradient);
  char* get_config_proto();
  void destroy();
  ~SGDOptimizer() {
    // clear memory by Tensor library
    delete momentums_;
  }
private:
  Tensor<T>* momentums_;
  double learning_rate;
  double momentum;
  double decay;
  bool nesterov;
};

template <class T>
class AdagradOptimizer : public ParameterOptimizer<T> {
public:
  void update(const Tensor<T> &gradient) {
  }
private:
  
};

template <class T>
class AdadeltaOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
    auto parameter = &(*parameter_.get());
    learning_rate = applyLinearLearningRate(config_);
    applyGradientDescent(parameter, gradient, learning_rate);
  }
private:
  
};

template <class T>
class AdamOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
    auto parameter = &(*parameter_.get());
    learning_rate = applyLinearLearningRate(config_);
    applyGradientDescent(parameter, gradient, learning_rate);
  }
private:
  
};

// template <class T>
// class MomentumOptimizer : public ParameterOptimizer {
// public:
//   /*! \brief call the applyXX for example  */
//   MomentumOptimizer(const paddle::optimizer_config &config);
//   void update(const Tensor<T> &gradient) {
//     learning_rate = applyExpLearningRate(config_);
//     applyMomentum(
//         parameter, gradient, momentum, learning_rate, mu, weight_decay);
//   }

// private:
//   double momentum;
// };

}  // namespace optimizer
}  // namespace paddle
#endif
