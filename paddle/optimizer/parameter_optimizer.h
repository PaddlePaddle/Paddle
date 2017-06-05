#ifndef PADDLE_PARAMETER_OPTIMIZER_H_
#define PADDLE_PARAMETER_OPTIMIZER_H_

#include <glog/logging.h>
#include <functional>
#include <string>
#include "OptimizerConfig.pb.h"
#include "Tensor.h"
#include "lr_policy.h"

namespace paddle {
namespace optimizer {

class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  // use config for pack trainig state
  ParameterOptimizer(const OptimizerConfig &config) : config_(config){};

  ParameterOptimizer(BaseLr *lr) : lr_policy(lr), num_sample_passed(0) {}
  virtual ~ParameterOptimizer() { delete parameter_; };

  static ParameterOptimizer *create(const ::std::string &config_proto);
  virtual void update(const Tensor *gradient) = 0;
  virtual real *get_weight() const;
  virtual void set_weight(Tensor *parameter);

public:
  OptimizerConfig config_;
  Tensor *parameter_;

  // learning rate policy
  BaseLr *lr_policy;
  uint64_t num_sample_passed;
};

}  // namespace optimizer
}  // namespace paddle

#endif
