#pragma once

#include <algorithm>
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

class LrPolicy {
public:
  virtual ~LrPolicy() {}
  virtual double LearningRate(const uint64_t num_sample_passed) = 0;
  virtual std::string SerializeState() = 0;
  virtual void DeserializeState(const std::string &state) = 0;
};

// constant learning rate policy
class ConstLr final : public LrPolicy {
public:
  ConstLr(double lr) : learning_rate_(lr){};
  double LearningRate(const uint64_t num_sample_passed) {
    return learning_rate_;
  }
  std::string SerializeState() {
    LrPolicyState state;
    state.set_learning_rate(learning_rate_);
    return state.SerializeAsString();
  }
  void DeserializeState(const std::string &str) {
    LrPolicyState state;
    state.ParseFromString(str);
    learning_rate_ = state.learning_rate();
  }

private:
  double learning_rate_;
};

class LinearLr final : public LrPolicy {
public:
  LinearLr(double lr, double lr_decay_a, double lr_decay_b)
      : learning_rate_(lr), lr_decay_a_(lr_decay_a), lr_decay_b_(lr_decay_b) {}
  double LearningRate(const uint64_t num_sample_passed) {
    return std::max(learning_rate_ - lr_decay_a_ * num_sample_passed,
                    lr_decay_b_);
  }
  std::string SerializeState() {
    LrPolicyState state;
    state.set_learning_rate(learning_rate_);
    state.set_lr_decay_a(lr_decay_a_);
    state.set_lr_decay_b(lr_decay_b_);
    return state.SerializeAsString();
  }
  void DeserializeState(const std::string &str) {
    LrPolicyState state;
    state.ParseFromString(str);
    learning_rate_ = state.learning_rate();
    lr_decay_a_ = state.lr_decay_a();
    lr_decay_b_ = state.lr_decay_b();
  }

private:
  double learning_rate_;
  double lr_decay_a_;
  double lr_decay_b_;
};

}  // namespace optimizer
}  // namespace paddle
