#pragma once

#include <algorithm>
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

class LrPolicy {
public:
  virtual ~LrPolicy() {}
  virtual double LearningRate(const uint64_t num_sample_passed) = 0;
};

// constant learning rate policy
class ConstLr final : public LrPolicy {
public:
  ConstLr(double lr) : learning_rate(lr){};
  double LearningRate(const uint64_t num_sample_passed) {
    return learning_rate;
  }

private:
  double learning_rate;
};

class LinearLr final : public LrPolicy {
public:
  LinearLr(double lr, double lr_decay_a, double lr_decay_b)
      : learning_rate(lr), lr_decay_a(lr_decay_a), lr_decay_b(lr_decay_b) {}
  double LearningRate(const uint64_t num_sample_passed) {
    return std::max(learning_rate - lr_decay_a * num_sample_passed, lr_decay_b);
  }

private:
  double learning_rate;
  double lr_decay_a;
  double lr_decay_b;
};

}  // namespace optimizer
}  // namespace paddle
