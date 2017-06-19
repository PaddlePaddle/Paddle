#pragma once

#include <algorithm>
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

class LrPolicy {
public:
  virtual ~LrPolicy() {}
  virtual double LearningRate(const uint64_t num_sample_passed) = 0;
  virtual const char *SerializeState(int *state_len) = 0;
  virtual void DeserializeState(const std::string &state) = 0;
};

// constant learning rate policy
class ConstLr final : public LrPolicy {
public:
  ConstLr(double lr) : learning_rate(lr){};
  double LearningRate(const uint64_t num_sample_passed) {
    return learning_rate;
  }
  const char *SerializeState(int *state_len) { return nullptr; }
  void DeserializeState(const std::string &state) {}

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
  const char *SerializeState(int *state_len) {
    // TODO(zhihong) : add lr_policy serialization
    return nullptr;
  }
  void DeserializeState(const std::string &state) {
    // TODO(zhihong) : add lr_policy serialization
  }

private:
  double learning_rate;
  double lr_decay_a;
  double lr_decay_b;
};

}  // namespace optimizer
}  // namespace paddle
