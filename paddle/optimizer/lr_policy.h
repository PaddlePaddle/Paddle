#ifndef PADDLE_OPTIMIZER_LR_POLICY_H_
#define PADDLE_OPTIMIZER_LR_POLICY_H_

#include <algorithm>
#include "OptimizerConfig.pb.h"

namespace paddle {
namespace optimizer {

class BaseLr {
public:
  BaseLr(double lr) : learning_rate(lr) {}
  virtual ~BaseLr() {}
  virtual double get_learning_rate(const uint64_t num_sample_passed) = 0;

protected:
  double learning_rate;
};

// constant learning rate policy
class ConstLr final : public BaseLr {
public:
  ConstLr(double lr) : BaseLr(lr){};
  double get_learning_rate(const uint64_t num_sample_passed) {
    return learning_rate;
  }
};

class LinearLr final : public BaseLr {
public:
  LinearLr(double lr, double lr_decay_a, double lr_decay_b)
      : BaseLr(lr), lr_decay_a(lr_decay_a), lr_decay_b(lr_decay_b) {}
  double get_learning_rate(const uint64_t num_sample_passed) {
    return std::max(learning_rate - lr_decay_a * num_sample_passed, lr_decay_b);
  }

private:
  double lr_decay_a;
  double lr_decay_b;
};

}  // namespace optimizer
}  // namespace paddle

#endif
