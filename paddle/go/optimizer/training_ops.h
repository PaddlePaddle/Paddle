#ifndef PADDLE_LIB_TRAINING_OPS_H_
#define PADDLE_LIB_TRAINING_OPS_H_
/*! \brief this file contains the optimizer algorithnm as ops
  name convention. applyXXX  e.g applyGradientDescent
 */

#include <algorithm>
#include "Tensor.h"

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

template <typename T>
void applyGradientDescent(const Tensor<T> &gradient);

template <typename T>
void applyMomentum(const Tensor<T> &gradient, double mu, double weight_decay);

template <typename T>
void applyAdam(const Tensor<T> &gradient, double mu, double learning_rate);
}  // namespace optimizer
}  // namespace paddle

#endif
