#ifndef PADDLE_OPTIMIZER_LR_POLICY_H_
#define PADDLE_OPTIMIZER_LR_POLICY_H_

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
  double get_learning_rate(const uint64_t num_sample_passed) {
    return learning_rate;
  }
};

}  // namespace optimizer
}  // namespace paddle

#endif
