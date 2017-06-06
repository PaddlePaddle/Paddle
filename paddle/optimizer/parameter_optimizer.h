#pragma once

#include <glog/logging.h>
#include <functional>
#include <string>
#include "OptimizerConfig.pb.h"
#include "lr_policy.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

const std::string kOptimizerVersion = "1.0";

class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  ParameterOptimizer(LrPolicy *lr) : lr_policy_(lr), num_sample_passed_(0) {}
  virtual ~ParameterOptimizer() { delete parameter_; };

  static ParameterOptimizer *Create(const std::string &config_proto);
  virtual const char *SerializeState();
  virtual void DeSerializeState(const std::string &state);
  virtual void Update(const Tensor *gradient) = 0;
  virtual real *get_weight() const;
  virtual void set_weight(Tensor *parameter);

protected:
  OptimizerConfig config_;
  Tensor *parameter_;

  // learning rate policy
  LrPolicy *lr_policy_;
  uint64_t num_sample_passed_;
};

}  // namespace optimizer
}  // namespace paddle
