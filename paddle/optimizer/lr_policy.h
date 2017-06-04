#ifndef PADDLE_OPTIMIZER_LR_POLICY_H_
#define PADDLE_OPTIMIZER_LR_POLICY_H_

#include "OptimizerConfig.ph.h"

namespace paddle {
namespace optimizer {

class BaseLr {
public:
  LrPolicyBase(const OpitmizerConfig &config) {
    learning_rate = config.lr_config().learning_rate();
  }
  virtual double get_learning_rate(const uint64_t num_sample_passed) = 0;
private:
  double learning_rate;
};

// constant learning rate policy
class ConstLr final : public BaseLr {
public:
  double get_learning_rate(const uint64_t num_sample_passed) {
    return learning_rate;
  }
};


}
}

#endif
